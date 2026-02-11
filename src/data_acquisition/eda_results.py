import os
import random
import csv
from collections import defaultdict, Counter
from pathlib import Path

from PIL import Image
import matplotlib.pyplot as plt

VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def list_images(folder: Path):
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXT])


def safe_open_image(path: Path):
    try:
        with Image.open(path) as im:
            im.load()
            return im.size  # (w, h)
    except Exception:
        return None


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def make_bar_chart(labels, values, title, out_path: Path):
    plt.figure()
    plt.bar(labels, values)
    plt.title(title)
    plt.xlabel("Clasă")
    plt.ylabel("Număr imagini")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def make_resolution_hist(resolutions, title, out_path: Path):
    # rezoluții = list of (w,h); facem histogramă pe lățime și înălțime separat
    widths = [w for w, h in resolutions]
    heights = [h for w, h in resolutions]

    plt.figure()
    plt.hist(widths, bins=20, alpha=0.7, label="width")
    plt.hist(heights, bins=20, alpha=0.7, label="height")
    plt.title(title)
    plt.xlabel("Pixeli")
    plt.ylabel("Frecvență")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def make_examples_grid(class_to_images, title, out_path: Path, per_class=3, thumb_size=(256, 256)):
    # class_to_images: dict[str, list[Path]]
    classes = sorted(class_to_images.keys())
    rows = len(classes)
    cols = per_class

    fig = plt.figure(figsize=(cols * 3.2, rows * 3.2))
    fig.suptitle(title)

    idx = 1
    for r, cls in enumerate(classes):
        picks = class_to_images[cls][:per_class]
        for c in range(cols):
            ax = fig.add_subplot(rows, cols, idx)
            ax.axis("off")
            if c < len(picks):
                p = picks[c]
                try:
                    with Image.open(p) as im:
                        im = im.convert("RGB")
                        im.thumbnail(thumb_size)
                        ax.imshow(im)
                        ax.set_title(f"{cls}: {p.name}", fontsize=7)
                except Exception:
                    ax.set_title(f"{cls}: [eroare la {p.name}]", fontsize=7)
            else:
                ax.set_title(f"{cls}: (lipsă)", fontsize=7)
            idx += 1

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=200)
    plt.close()


def run_eda(data_root: Path, results_root: Path, splits=("train", "validation", "test"), include_processed=False, seed=42):
    random.seed(seed)

    eda_dir = results_root / "eda"
    ensure_dir(eda_dir)

    # colectare statistici
    counts = []  # rows for CSV
    global_summary = []

    for split in splits:
        split_path = data_root / split
        if not split_path.exists():
            global_summary.append(f"[WARN] Splitul '{split}' nu există: {split_path}")
            continue

        class_dirs = sorted([p for p in split_path.iterdir() if p.is_dir()])
        if not class_dirs:
            global_summary.append(f"[WARN] Nicio clasă găsită în: {split_path}")
            continue

        split_class_counts = {}
        split_resolutions = []
        corrupted = 0

        # pentru colaj
        examples = {}

        for cls_dir in class_dirs:
            cls = cls_dir.name
            imgs = list_images(cls_dir)
            split_class_counts[cls] = len(imgs)

            # sample pentru grid
            sample = imgs[:]
            random.shuffle(sample)
            examples[cls] = sample[:3]

            # rezoluții (sample, ca să nu fie prea lent pe dataset mare)
            # ia până la 300 imagini per clasă pentru rezoluții
            for p in imgs[:300]:
                size = safe_open_image(p)
                if size is None:
                    corrupted += 1
                else:
                    split_resolutions.append(size)

            counts.append({"split": split, "class": cls, "n_images": len(imgs)})

        # salvare chart distribuție clase
        labels = list(split_class_counts.keys())
        values = [split_class_counts[k] for k in labels]
        make_bar_chart(labels, values, f"Distribuția imaginilor pe clase - {split}", eda_dir / f"class_distribution_{split}.png")

        # salvare rezoluții
        if split_resolutions:
            make_resolution_hist(split_resolutions, f"Distribuția rezoluțiilor (sample) - {split}", eda_dir / f"resolution_distribution_{split}.png")

        # colaj exemple
        make_examples_grid(examples, f"Exemple imagini - {split}", eda_dir / f"examples_{split}.png", per_class=3)

        # rezumat split
        total = sum(split_class_counts.values())
        global_summary.append(f"{split}: total={total} | pe clase={split_class_counts} | imagini corupte detectate (sample)={corrupted}")

    # opțional: processed
    if include_processed:
        proc_path = data_root / "processed"
        if proc_path.exists():
            class_dirs = sorted([p for p in proc_path.iterdir() if p.is_dir()])
            if class_dirs:
                proc_counts = {d.name: len(list_images(d)) for d in class_dirs}
                make_bar_chart(list(proc_counts.keys()), list(proc_counts.values()),
                               "Distribuția imaginilor pe clase - processed",
                               eda_dir / "class_distribution_processed.png")
                global_summary.append(f"processed: pe clase={proc_counts}")
            else:
                global_summary.append("[WARN] 'processed' există dar nu are subfoldere de clase.")
        else:
            global_summary.append("[WARN] include_processed=True, dar 'data/processed' nu există.")

    # scrie counts.csv
    csv_path = eda_dir / "counts.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "class", "n_images"])
        writer.writeheader()
        writer.writerows(counts)

    # scrie summary.txt
    with open(eda_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("EDA Summary\n")
        f.write("===========\n\n")
        for line in global_summary:
            f.write(line + "\n")

    print(f"[OK] EDA generat în: {eda_dir}")
    print(f" - {csv_path}")
    print(f" - {eda_dir / 'summary.txt'}")


if __name__ == "__main__":
    # ✅ MODIFICĂ DOAR CALEA CĂTRE PROIECTUL TĂU
    PROJECT_ROOT = Path(r"C:\Users\DANI\OneDrive\Desktop\Proiect_Final_Rada_Andrei_Daniel_Rn")
    DATA_ROOT = PROJECT_ROOT / "data"
    RESULTS_ROOT = PROJECT_ROOT / "results"

    run_eda(
        data_root=DATA_ROOT,
        results_root=RESULTS_ROOT,
        splits=("train", "validation", "test"),
        include_processed=False,   # pune True dacă vrei și pentru processed
        seed=42
    )
