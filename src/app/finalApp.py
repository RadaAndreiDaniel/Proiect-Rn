# =========================
# PATH SETUP (IMPORTANT)
# =========================
import sys
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parents[1]
ROOT_PATH = SRC_PATH.parents[0]

sys.path.append(str(SRC_PATH))

# =========================
# IMPORTS
# =========================
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2

from data_acquisition.annotator import Annotator

# =========================
# CONFIG
# =========================
MODEL_PATH = ROOT_PATH / "models" / "trained_model.pth"
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = ["iarna", "mixt", "vara"]

# =========================
# LOAD MODEL (CREIER 🧠)
# =========================
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 3)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    model.to(DEVICE)

    return model

# =========================
# PREPROCESS IMAGE
# =========================
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(DEVICE)

# =========================
# PIPELINE FINAL
# =========================
def run_pipeline(image_path):

    image_path = Path(image_path)

    # ---- LOAD IMAGE (GRAYSCALE) ----
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError("Nu pot citi imaginea!")

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ---- MANUAL ANNOTATION ----
    annotator = Annotator(img, max_width=1800, max_height=800)
    result = annotator.annotate_keypoints()

    if not result:
        print("❌ Adnotare anulată.")
        return

    # ---- SAVE PREVIEW 256×256 ----
    preview = annotator._last_preview
    preview_resized = cv2.resize(preview, (256, 256))

    preview_path = image_path.parent / f"{image_path.stem}_preview_256.jpg"
    cv2.imwrite(str(preview_path), preview_resized)

    print(f"✔ Preview salvat: {preview_path}")

    # ---- LOAD MODEL ----
    model = load_model()

    # ---- PREDICT ----
    input_tensor = preprocess_image(preview_path)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    predicted_idx = torch.argmax(probs).item()

    # ---- RESULTS ----
    print("\n🧠 VERDICT FINAL")
    print("----------------------------")
    print(f"Tip anvelopă: {CLASS_NAMES[predicted_idx]}\n")

    print("Confidence pe clase:")
    for cls, prob in zip(CLASS_NAMES, probs):
        print(f"  {cls}: {prob.item() * 100:.2f}%")

# =========================
# CLI
# =========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Adnotare manuală + clasificare bandă de rulare"
    )
    parser.add_argument(
        "-i", "--image", required=True, help="Cale către imaginea de test"
    )

    args = parser.parse_args()
    run_pipeline(args.image)
