import cv2
import os
from pathlib import Path
import numpy as np

# =========================
# CONFIG
# =========================
RAW_DIR = Path("../../data/processed")
PROCESSED_DIR = Path("../../data/processed")

IMG_SIZE = 224
USE_GRAYSCALE = True
USE_CLAHE = True
USE_BLUR = False  # blur usor, optional

CLASSES = ["iarna", "mixt", "vara"]

# =========================
# CREATE FOLDERS
# =========================
for cls in CLASSES:
    (PROCESSED_DIR / cls).mkdir(parents=True, exist_ok=True)

# =========================
# PREPROCESS FUNCTION
# =========================
def preprocess_image(img):
    # Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Grayscale
    if USE_GRAYSCALE:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # CLAHE (contrast local)
        if USE_CLAHE:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe.apply(img)

        # back to 3 channels (pentru ResNet)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Blur usor (optional)
    if USE_BLUR:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    return img

# =========================
# PROCESS DATASET
# =========================
for cls in CLASSES:
    input_dir = RAW_DIR / cls
    output_dir = PROCESSED_DIR / cls

    for img_name in os.listdir(input_dir):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = input_dir / img_name
        img = cv2.imread(str(img_path))

        if img is None:
            print(f"[SKIP] {img_path}")
            continue

        processed = preprocess_image(img)

        out_path = output_dir / img_name
        cv2.imwrite(str(out_path), processed)

    print(f"[OK] {cls} procesat")

print("✅ Dataset procesat complet")
