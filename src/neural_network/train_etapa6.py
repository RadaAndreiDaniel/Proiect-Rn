"""
Etapa 6 – Train + Evaluate (ResNet18) pe folderele: train / validation / test
- Citește datele din locația ta Windows (DATA_DIR)
- Salvează toate artefactele cu denumiri „de Etapa 6” și sufix _final

IMPORTANT:
Structura așteptată în DATA_DIR:
DATA_DIR/
  train/
    vara/ iarna/ mixt/
  validation/   (sau val / valid)
    vara/ iarna/ mixt/
  test/
    vara/ iarna/ mixt/

Exemple de rulare:
  python train_etapa6_final.py

Dacă vrei alt path:
  python train_etapa6_final.py --data_dir "C:\\Users\\DANI\\OneDrive\\Desktop\\Proiect_Final_Rada_Andrei_Daniel_Rn\\data"
"""

import os
import json
import csv
import shutil
import argparse
from pathlib import Path
from multiprocessing import freeze_support
from typing import Optional

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
# DEFAULT PATH (Windows)
# =========================
DEFAULT_DATA_DIR = r"C:\Users\DANI\Desktop\Proiect_Final_Rada_Andrei_Daniel_Rn\data"


# =========================
# Helpers: val folder
# =========================
def find_val_dir(root: Path) -> Optional[Path]:
    for name in ["validation", "val", "valid"]:
        p = root / name
        if p.exists():
            return p
    return None


def ensure_dirs(data_dir: Path):
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    val_dir = find_val_dir(data_dir)

    assert train_dir.exists(), f"Lipsește: {train_dir}"
    assert test_dir.exists(), f"Lipsește: {test_dir}"
    assert val_dir is not None and val_dir.exists(), f"Lipsește validation/val/valid în: {data_dir}"

    return train_dir, val_dir, test_dir


# =========================
# Industrial-like augmentations
# =========================
class AddGaussianNoise:
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


class RandomGaussianBlurPIL:
    def __init__(self, radius_min=0.0, radius_max=1.2, p=0.25):
        self.radius_min = radius_min
        self.radius_max = radius_max
        self.p = p

    def __call__(self, img: Image.Image):
        if np.random.rand() > self.p:
            return img
        r = float(np.random.uniform(self.radius_min, self.radius_max))
        return img.filter(ImageFilter.GaussianBlur(radius=r))


def build_transforms(img_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomPerspective(distortion_scale=0.18, p=0.6),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.10, hue=0.02),
        RandomGaussianBlurPIL(radius_min=0.0, radius_max=1.2, p=0.25),
        transforms.ToTensor(),
        AddGaussianNoise(mean=0.0, std=0.02, p=0.50),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return train_tf, eval_tf


@torch.no_grad()
def evaluate(model, loader, criterion, device: str):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

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
    plt.title("Learning Curves (Loss / Val Loss)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_confusion_matrix(cm, class_names, out_path: Path):
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix (Optimized)")
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
    conf = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                conf.append((int(cm[i, j]), class_names[i], class_names[j]))
    conf.sort(reverse=True, key=lambda x: x[0])

    lines = []
    lines.append("# Analiză erori – context industrial (Etapa 6)\n\n")
    lines.append("## Confuzii principale (true → predicted)\n")
    if conf:
        for n, t, p in conf[:8]:
            lines.append(f"- **{t} → {p}**: {n} cazuri\n")
    else:
        lines.append("- Nu există confuzii semnificative pe test (posibil test prea mic).\n")

    lines.append("\n## Interpretare\n")
    lines.append(
        "- Erorile apar frecvent când iluminarea este neuniformă, există blur (vibrații) sau ROI nu surprinde complet banda de rulare.\n"
        "- În special, clasele **mixt** și **iarnă** pot fi similare vizual în anumite condiții.\n"
    )

    lines.append("\n## Recomandări\n")
    lines.append(
        "- Creșterea numărului de exemple originale pentru clasa sub-reprezentată.\n"
        "- Augmentări specifice domeniului (brightness/contrast, blur, perspective) – deja incluse la train.\n"
        "- Folosirea unui prag de încredere (softmax) în UI pentru cazurile incerte.\n"
    )

    lines.append("\n## Classification report (test)\n")
    lines.append("```\n")
    lines.append(report_text.strip() + "\n")
    lines.append("```\n")

    out_path.write_text("".join(lines), encoding="utf-8")


def export_misclassified_images(dataset, y_true, y_pred, out_dir: Path, max_copy=30):
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
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


def save_training_history_csv(history, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["epoch", "loss", "val_loss", "val_accuracy", "val_f1_macro", "lr"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for h in history:
            w.writerow({k: h.get(k, "") for k in fieldnames})


def save_optimization_experiments_csv(exp_row: dict, out_path: Path):
    """
    Etapa 6 cere un tabel comparativ de experimente. Dacă rulezi doar un model final,
    măcar salvezi acest CSV cu o singură intrare ("Exp 1").
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["exp", "description", "accuracy", "f1_macro", "epochs_ran", "batch_size", "lr", "notes"]
    write_header = not out_path.exists()
    with out_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow({k: exp_row.get(k, "") for k in fieldnames})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Folder care conține train/validation/test")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--scheduler_patience", type=int, default=2)
    args = parser.parse_args()

    # =========================
    # OUTPUTS (Etapa 6 + _final)
    # =========================
    project_root = Path(__file__).resolve().parents[1]  # ajustează dacă pui fișierul în altă locație
    models_dir = (project_root / "models").resolve()
    results_dir = (project_root / "results").resolve()
    docs_dir = (project_root / "docs").resolve()
    docs_results_dir = docs_dir / "results"

    for d in [models_dir, results_dir, docs_dir, docs_results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Etapa 6: nume cerute + sufix _final
    MODEL_OUT = models_dir / "optimized_model_final.pt"
    FINAL_METRICS_OUT = results_dir / "final_metrics_final.json"
    OPT_EXPERIMENTS_OUT = results_dir / "optimization_experiments_final.csv"
    TRAINING_HISTORY_OUT = results_dir / "training_history_final.csv"

    CM_OUT = docs_dir / "confusion_matrix_optimized_final.png"
    LEARNING_CURVES_OUT = docs_results_dir / "learning_curves_final_final.png"
    ERROR_ANALYSIS_OUT = docs_dir / "error_analysis_final.md"
    ERRORS_DIR = docs_dir / "errors_final"

    # =========================
    # DEVICE
    # =========================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory = (device == "cuda")
    num_workers = 0  # Windows safe

    # =========================
    # DATA
    # =========================
    data_dir = Path(args.data_dir).resolve()
    train_dir, val_dir, test_dir = ensure_dirs(data_dir)

    print("[INFO] Device:", device)
    print("[INFO] DATA_DIR:", data_dir)
    print("[INFO] Train:", train_dir)
    print("[INFO] Val:", val_dir)
    print("[INFO] Test:", test_dir)

    train_tf, eval_tf = build_transforms(args.img_size)

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_tf)
    test_ds = datasets.ImageFolder(test_dir, transform=eval_tf)

    class_names = train_ds.classes
    num_classes = len(class_names)

    print("[INFO] Classes:", class_names)
    print("[INFO] Sizes: train", len(train_ds), "val", len(val_ds), "test", len(test_ds))

    # WeightedRandomSampler (ajută macro F1)
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

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    # =========================
    # MODEL
    # =========================
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.scheduler_patience
    )

    best_val_loss = float("inf")
    best_epoch = -1
    patience_left = args.early_stop_patience

    history = []

    # =========================
    # TRAIN
    # =========================
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        train_loss = float(np.mean(losses)) if losses else 0.0

        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)
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
            f"Epoch {epoch}/{args.epochs} - "
            f"loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - "
            f"val_acc: {val_acc:.4f} - val_f1: {val_f1:.4f} - lr: {lr_now:.2e}"
        )

        # Best checkpoint on val_loss
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_left = args.early_stop_patience

            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "best_val_loss": float(best_val_loss),
                "class_names": class_names,
                "img_size": args.img_size,
                "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            }, MODEL_OUT)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[EARLY STOP] val_loss nu a scăzut {args.early_stop_patience} epoci consecutive. Best epoch={best_epoch}")
                break

    # Save learning curves + history CSV (Etapa 6)
    plot_loss_curve(history, LEARNING_CURVES_OUT)
    save_training_history_csv(history, TRAINING_HISTORY_OUT)
    print(f"[OK] Saved learning curves: {LEARNING_CURVES_OUT}")
    print(f"[OK] Saved training history CSV: {TRAINING_HISTORY_OUT}")

    # =========================
    # TEST (best model)
    # =========================
    ckpt = torch.load(MODEL_OUT, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_loss, test_acc, test_f1, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    report_txt = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    save_confusion_matrix(cm, class_names, CM_OUT)
    print(f"[OK] Saved confusion matrix: {CM_OUT}")

    # Save final metrics JSON (Etapa 6)
    metrics = {
        "model": str(MODEL_OUT.name),
        "best_epoch": int(ckpt.get("epoch", -1)),
        "best_val_loss": float(ckpt.get("best_val_loss", float("nan"))),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "test_f1_macro": float(test_f1),
        "classes": class_names,
        "classification_report_text": report_txt,
        "confusion_matrix": cm.tolist(),
        "history": history
    }
    FINAL_METRICS_OUT.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[OK] Saved final metrics: {FINAL_METRICS_OUT}")

    # Save error analysis markdown
    write_error_analysis(cm, class_names, report_txt, ERROR_ANALYSIS_OUT)
    print(f"[OK] Saved error analysis: {ERROR_ANALYSIS_OUT}")

    # Export some misclassified images (optional, but useful)
    try:
        export_misclassified_images(test_ds, y_true, y_pred, ERRORS_DIR, max_copy=30)
        print(f"[OK] Exported misclassified samples: {ERRORS_DIR}")
    except Exception:
        print("[WARN] Could not export misclassified images (not critical).")

    # Save experiments CSV (Etapa 6)
    exp_row = {
        "exp": "Exp_1",
        "description": "ResNet18 + augmentări industriale + sampler ponderat + ReduceLROnPlateau + EarlyStopping",
        "accuracy": f"{test_acc:.4f}",
        "f1_macro": f"{test_f1:.4f}",
        "epochs_ran": str(best_epoch),
        "batch_size": str(args.batch_size),
        "lr": str(args.lr),
        "notes": "Model salvat ca optimized_model_final.pt; metrici în final_metrics_final.json"
    }
    save_optimization_experiments_csv(exp_row, OPT_EXPERIMENTS_OUT)
    print(f"[OK] Saved optimization experiments: {OPT_EXPERIMENTS_OUT}")

    print("\n===== TEST RESULTS (BEST MODEL) =====")
    print(f"Test Accuracy:   {test_acc:.4f}")
    print(f"Test F1 (macro): {test_f1:.4f}")
    print("\n===== CLASSIFICATION REPORT =====")
    print(report_txt)


if __name__ == "__main__":
    freeze_support()
    main()
