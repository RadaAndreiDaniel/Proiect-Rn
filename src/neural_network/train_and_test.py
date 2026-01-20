import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import csv
from pathlib import Path

# =========================
# CONFIG
# =========================
DATA_DIR = "../../data/processed"
MODEL_PATH = "../../models/trained_model_final.pth"
RESULTS_DIR = Path("../../results")

BATCH_SIZE = 16
EPOCHS = 10
LR = 0.001
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = ["iarna", "mixt", "vara"]

RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# =========================
# TRANSFORMS
# =========================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# DATASET + STRATIFIED SPLIT
# =========================
full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

targets = np.array(full_dataset.targets)
indices = np.arange(len(targets))

# 70% train, 30% temp
train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.30,
    stratify=targets,
    random_state=42
)

# 15% val, 15% tests
val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    stratify=targets[temp_idx],
    random_state=42
)

train_dataset = Subset(full_dataset, train_idx)
val_dataset   = Subset(full_dataset, val_idx)
test_dataset  = Subset(full_dataset, test_idx)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# =========================
# MODEL
# =========================
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# =========================
# TRAINING HISTORY
# =========================
history = []

# =========================
# TRAIN + VALIDATION
# =========================
for epoch in range(EPOCHS):
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_correct += (outputs.argmax(1) == labels).sum().item()
        train_total += labels.size(0)

    train_loss /= len(train_loader)
    train_acc = train_correct / train_total

    # VALIDATION
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= len(val_loader)
    val_acc = val_correct / val_total

    history.append([epoch + 1, train_loss, train_acc, val_loss, val_acc])

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | "
        f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
    )

# =========================
# TEST EVALUATION (OFICIAL)
# =========================
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = accuracy_score(all_labels, all_preds)
test_f1  = f1_score(all_labels, all_preds, average="macro")

print("\n===== TEST SET RESULTS =====")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test F1-score (macro): {test_f1:.4f}")

# =========================
# SAVE MODEL
# =========================
torch.save(model.state_dict(), MODEL_PATH)

# =========================
# SAVE CSV
# =========================
csv_path = RESULTS_DIR / "training_history.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy"
    ])
    writer.writerows(history)

print(f"[OK] Model salvat: {MODEL_PATH}")
print(f"[OK] Training history salvat: {csv_path}")
