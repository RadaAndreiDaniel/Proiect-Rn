from pathlib import Path
import random
import shutil

PROCESSED_PATH = Path("data/processed")
TRAIN_PATH = Path("data/train")
VAL_PATH = Path("data/validation")
TEST_PATH = Path("data/test")

SPLIT = (0.7, 0.15, 0.15)
RANDOM_SEED = 42

def main():
    random.seed(RANDOM_SEED)

    for p in [TRAIN_PATH, VAL_PATH, TEST_PATH]:
        p.mkdir(parents=True, exist_ok=True)

    for class_dir in PROCESSED_PATH.iterdir():
        if not class_dir.is_dir():
            continue

        images = list(class_dir.glob("*.*"))
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * SPLIT[0])
        n_val = int(n_total * SPLIT[1])

        splits = {
            TRAIN_PATH / class_dir.name: images[:n_train],
            VAL_PATH / class_dir.name: images[n_train:n_train+n_val],
            TEST_PATH / class_dir.name: images[n_train+n_val:]
        }

        for split_dir, imgs in splits.items():
            split_dir.mkdir(parents=True, exist_ok=True)
            for img in imgs:
                shutil.copy(img, split_dir / img.name)

    print("✓ Split train / validation / test finalizat")

if __name__ == "__main__":
    main()
