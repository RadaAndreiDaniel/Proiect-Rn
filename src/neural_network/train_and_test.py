import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from PIL import Image

# =========================
# CONFIG
# =========================
DATA_DIR = "../../data/processed"
TEST_IMAGE = "../../data/test/test_image7.jpg"
MODEL_PATH = "../../models/trained_model.pth"

BATCH_SIZE = 16
EPOCHS = 10
LR = 0.001
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = ["iarna", "mixt", "vara"]

# =========================
# TRANSFORMS
# =========================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

# =========================
# DATASET + SPLIT
# =========================
full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# =========================
# MODEL
# =========================
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 3)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# =========================
# TRAIN
# =========================
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {running_loss/len(train_loader):.4f}")

# =========================
# SAVE MODEL (CREIERUL 🧠)
# =========================
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model antrenat salvat în {MODEL_PATH}")

# =========================
# TEST ONE IMAGE
# =========================
model.eval()

image = Image.open(TEST_IMAGE).convert("RGB")
image = transforms.Resize((IMG_SIZE, IMG_SIZE))(image)
image = transforms.ToTensor()(image).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)

print(f"Imaginea de test este clasificată ca: {CLASS_NAMES[predicted.item()]}")
