import os
import json
import shutil
from pathlib import Path
from multiprocessing import freeze_support

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from torchvision import datasets, transforms, models
from PIL import Image, ImageFilter

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt


# =========================
# CONFIG
# =========================
DATA_DIR = Path("../../data/processed").resolve()  # contine train/ val(or valid)/ test/
MODELS_DIR = Path("../../models").resolve()
RESULTS_DIR = Path("../../results").resolve()
DOCS_DIR = Path("../../docs").resolve()

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_OUT = MODELS_DIR / "trained_model.pt"              # creierul (checkpoint)
TEST_METRICS_OUT = RESULTS_DIR / "test_metrics.json"     # cerinta
CM_OUT = DOCS_DIR / "confusion_matrix.png"               # cerinta
LOSS_CURVE_OUT = DOCS_DIR / "loss_curve.png"             # cerinta
ERROR_ANALYSIS_OUT = DOCS_DIR / "error_analysis.md"      # cerinta
ERRORS_DIR = DOCS_DIR / "errors"                         # optional (imagini gresite)

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LR = 3e-4
WEIGHT_DECAY = 1e-4

# Nivel 2:
EARLY_STOPPING_PATIENCE = 5  # val_loss nu scade 5 epoci consecutive
SCHEDULER_PATIENCE = 2

# Windows safe
NUM_WORKERS = 0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = (DEVICE == "cuda")


# =========================
# PATHS: train/val/test
# =========================
def find_val_dir(root: Path):
    for name in ["val", "valid", "validation"]:
        p = root / name
        if p.exists():
            return p
    return None


def ensure_dirs(data_dir: Path):
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    val_dir = find_val_dir(data_dir)

    assert train_dir.exists(), f"Lipseste: {train_dir}"
    assert test_dir.exists(), f"Lipseste: {test_dir}"
    assert val_dir is not None and val_dir.exists(), f"Lipseste val/valid/validation in: {data_dir}"

    return train_dir, val_dir, test_dir


# =========================
# Industrial-like augmentations (fara Albumentations)
# - slight perspective
# - lighting variation
# - blur mic (vibratii)
# - noise mic (gaussian) + compresie/artefacte usoare (simulat)
# FARA rotatii simple mari
# =========================
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.02, p=0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, tensor):
        if torch.rand(1).item() > self.p:
            return tensor
        noise = torch.randn_like(tensor) * self.std + self.mean
        out = tensor + noise
        return torch.clamp(out, 0.0, 1.0)


class RandomGaussianBlurPIL(object):
    def __init__(self, radius_min=0.0, radius_max=1.2, p=0.25):
        self.radius_min = radius_min
        self.radius_max = radius_max
        self.p = p

    def __call__(self, img: Image.Image):
        if np.random.rand() > self.p:
            return img
        r = float(np.random.uniform(self.radius_min, self.radius_max))
        return img.filter(ImageFilter.GaussianBlur(radius=r))


def build_transforms():
    # Train: perspective + lighting + blur + noise (industrial)
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomPerspective(distortion_scale=0.18, p=0.6),  # slight perspective
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.10, hue=0.02),  # lighting
        RandomGaussianBlurPIL(radius_min=0.0, radius_max=1.2, p=0.25),  # blur mic
        transforms.ToTensor(),
        AddGaussianNoise(mean=0.0, std=0.02, p=0.50),  # noise mic
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Eval: strict ca in productie
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return train_tf, eval_tf


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += float(loss.item())

        preds = logits.argmax(1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / max(len(loader), 1)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, f1, np.array(all_labels), np.array(all_preds)


def plot_loss_curve(history, out_path: Path):
    epochs = [h["epoch"] for h in history]
    loss = [h["loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]

    plt.figure()
    plt.plot(epochs, loss, label="loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss & Val Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_confusion_matrix(cm, class_names, out_path: Path):
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(class_names)), class_names, rotation=30, ha="right")
    plt.yticks(range(len(class_names)), class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def write_error_analysis(cm, class_names, report_text, out_path: Path):
    # top confuzii off-diagonal
    conf = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                conf.append((int(cm[i, j]), class_names[i], class_names[j]))
    conf.sort(reverse=True, key=lambda x: x[0])

    lines = []
    lines.append("# Analiză erori – context industrial (Nivel 2)\n\n")
    lines.append("## Observații cheie\n")
    if conf:
        lines.append("Cele mai frecvente confuzii (true → predicted):\n")
        for n, t, p in conf[:5]:
            lines.append(f"- **{t} → {p}**: {n} cazuri\n")
    else:
        lines.append("- Nu există confuzii semnificative pe test (posibil test prea mic).\n")

    lines.append("\n## Interpretare industrială\n")
    lines.append(
        "- În medii industriale, **iluminarea variabilă**, reflexiile și **vibrațiile** produc blur/noise care schimbă textura.\n"
        "- Variațiile de **perspectivă** (unghi camera / poziționare) pot altera pattern-ul perceput.\n"
        "- Augmentările folosite simulează aceste efecte (perspective + lighting + blur + noise) pentru a crește robustețea.\n"
    )

    lines.append("## Recomandări tehnice\n")
    lines.append(
        "- Creșteți setul de **test** (19 imagini e foarte puțin) pentru scoruri stabile.\n"
        "- Mențineți echilibrarea claselor (sampler ponderat inclus).\n"
        "- Pentru aplicație reală, utilizați prag de încredere (softmax) și tratați cazurile incerte.\n"
    )

    lines.append("\n## Raport clasificare (test)\n")
    lines.append("```\n")
    lines.append(report_text.strip() + "\n")
    lines.append("```\n")

    out_path.write_text("".join(lines), encoding="utf-8")


def export_misclassified_images(dataset, y_true, y_pred, out_dir: Path, max_copy=30):
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = 0

    # dataset.samples: (path, label)
    for idx, (t, p) in enumerate(zip(y_true, y_pred)):
        if t == p:
            continue

        src_path, true_lbl = dataset.samples[idx]
        src = Path(src_path)
        dst = out_dir / f"true_{dataset.classes[true_lbl]}__pred_{dataset.classes[p]}__{src.name}"
        try:
            shutil.copy2(src, dst)
            copied += 1
        except Exception:
            pass

        if copied >= max_copy:
            break


def main():
    train_dir, val_dir, test_dir = ensure_dirs(DATA_DIR)

    print("[INFO] Device:", DEVICE)
    print("[INFO] DATA_DIR:", DATA_DIR)
    print("[INFO] Train:", train_dir)
    print("[INFO] Val:", val_dir)
    print("[INFO] Test:", test_dir)

    train_tf, eval_tf = build_transforms()

    # Datasets
    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_tf)
    test_ds = datasets.ImageFolder(test_dir, transform=eval_tf)

    class_names = train_ds.classes
    num_classes = len(class_names)

    print("[INFO] Classes:", class_names)
    print("[INFO] Sizes: train", len(train_ds), "val", len(val_ds), "test", len(test_ds))

    # WeightedRandomSampler pentru echilibrare (ajuta macro F1)
    targets = np.array(train_ds.targets)
    class_counts = np.bincount(targets, minlength=num_classes).astype(np.float32)
    class_weights = 1.0 / np.maximum(class_counts, 1.0)
    sample_weights = class_weights[targets]

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True
    )

    print("[INFO] Train class counts:", dict(zip(class_names, class_counts.astype(int))))

    # Loaders (Windows safe)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # Model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Nivel 2: ReduceLROnPlateau pe val_loss
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=SCHEDULER_PATIENCE
    )

    # Early stopping pe val_loss (patience=5)
    best_val_loss = float("inf")
    best_epoch = -1
    patience_left = EARLY_STOPPING_PATIENCE

    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            losses.append(float(loss.item()))

        train_loss = float(np.mean(losses)) if losses else 0.0

        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        lr_now = float(optimizer.param_groups[0]["lr"])
        history.append({
            "epoch": epoch,
            "loss": train_loss,
            "val_loss": float(val_loss),
            "val_accuracy": float(val_acc),
            "val_f1_macro": float(val_f1),
            "lr": lr_now
        })

        print(
            f"Epoch {epoch}/{EPOCHS} - "
            f"loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - "
            f"val_acc: {val_acc:.4f} - val_f1: {val_f1:.4f} - lr: {lr_now:.2e}"
        )

        # Best model on val_loss
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_left = EARLY_STOPPING_PATIENCE

            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "best_val_loss": float(best_val_loss),
                "class_names": class_names,
                "img_size": IMG_SIZE,
                "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            }, MODEL_OUT)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[EARLY STOP] val_loss nu a scazut 5 epoci consecutive. Best epoch={best_epoch}")
                break

    # Loss curve (cerinta)
    plot_loss_curve(history, LOSS_CURVE_OUT)
    print(f"[OK] Saved loss curve: {LOSS_CURVE_OUT}")

    # Load best model and evaluate on TEST
    ckpt = torch.load(MODEL_OUT, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_loss, test_acc, test_f1, y_true, y_pred = evaluate(model, test_loader, criterion)

    report_txt = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    # Save confusion matrix (cerinta)
    save_confusion_matrix(cm, class_names, CM_OUT)
    print(f"[OK] Saved confusion matrix: {CM_OUT}")

    # Save metrics json (cerinta)
    metrics = {
        "best_epoch": int(ckpt.get("epoch", -1)),
        "best_val_loss": float(ckpt.get("best_val_loss", float("nan"))),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "test_f1_macro": float(test_f1),
        "classes": class_names,
        "classification_report_text": report_txt,
        "confusion_matrix": cm.tolist(),
        "history": history,
    }
    TEST_METRICS_OUT.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[OK] Saved test metrics: {TEST_METRICS_OUT}")

    # Error analysis (cerinta)
    write_error_analysis(cm, class_names, report_txt, ERROR_ANALYSIS_OUT)
    print(f"[OK] Saved error analysis: {ERROR_ANALYSIS_OUT}")

    # Optional: export some misclassified
    try:
        export_misclassified_images(test_ds, y_true, y_pred, ERRORS_DIR, max_copy=30)
        print(f"[OK] Exported misclassified samples: {ERRORS_DIR}")
    except Exception:
        print("[WARN] Could not export misclassified images (not critical).")

    print("\n===== TEST RESULTS (BEST MODEL) =====")
    print(f"Test Accuracy:   {test_acc:.4f}")
    print(f"Test F1 (macro): {test_f1:.4f}")
    print("\n===== CLASSIFICATION REPORT =====")
    print(report_txt)


if __name__ == "__main__":
    freeze_support()
    main()
