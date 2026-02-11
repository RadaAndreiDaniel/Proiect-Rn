import numpy as np
from pathlib import Path
from PIL import Image

PROCESSED_PATH = Path("data/processed")

def main():
    for class_dir in PROCESSED_PATH.iterdir():
        if not class_dir.is_dir():
            continue

        for img_path in class_dir.glob("*.*"):
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img).astype("float32") / 255.0  # normalizare [0,1]
            img_norm = Image.fromarray((img_np * 255).astype("uint8"))
            img_norm.save(img_path)

    print("✓ Normalizare pixeli [0,1] aplicată")

if __name__ == "__main__":
    main()
