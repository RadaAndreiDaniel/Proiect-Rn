import hashlib
from pathlib import Path
from PIL import Image
import shutil

RAW_PATH = Path("data/raw")
PROCESSED_PATH = Path("data/processed")
IMG_SIZE = (224, 224)

def hash_image(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def main():
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    seen_hashes = set()

    for class_dir in RAW_PATH.iterdir():
        if not class_dir.is_dir():
            continue

        output_class_dir = PROCESSED_PATH / class_dir.name
        output_class_dir.mkdir(parents=True, exist_ok=True)

        for img_path in class_dir.glob("*.*"):
            try:
                img_hash = hash_image(img_path)
                if img_hash in seen_hashes:
                    continue

                img = Image.open(img_path).convert("RGB")
                img = img.resize(IMG_SIZE)

                save_path = output_class_dir / img_path.name
                img.save(save_path)

                seen_hashes.add(img_hash)

            except Exception as e:
                print(f"Eroare {img_path.name}: {e}")

    print("✓ Duplicate eliminate + resize aplicat")

if __name__ == "__main__":
    main()
