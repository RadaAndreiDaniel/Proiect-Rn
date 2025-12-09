import os
import cv2
from ultralytics import YOLO
from extract_tire_pattern import extract_tire_pattern

INPUT_ROOT = "data/train"
OUTPUT_ROOT = "data/processed/train_patterns"
YOLO_MODEL_PATH = "yolov8n.pt"
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_images(root):
    items = []
    for cls in os.listdir(root):
        cls_folder = os.path.join(root, cls)
        if not os.path.isdir(cls_folder):
            continue
        for f in os.listdir(cls_folder):
            if f.lower().endswith(VALID_EXT):
                items.append((cls, os.path.join(cls_folder, f)))
    return items


def detect_wheel_bbox(model, img):
    """Returnează bounding box-ul roții detectate, sau None dacă nu găsește."""
    results = model(img)
    best_box = None
    best_conf = 0.0

    for r in results:
        for box in r.boxes:
            conf = float(box.conf)
            if conf > best_conf:
                best_conf = conf
                best_box = box

    if best_box is None:
        return None

    x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
    return x1, y1, x2, y2


def process_dataset():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    print("[INFO] Încarc modelul YOLO pentru detectarea roților...")
    model = YOLO(YOLO_MODEL_PATH)

    items = list_images(INPUT_ROOT)
    print(f"[INFO] Găsit {len(items)} imagini în {INPUT_ROOT}")

    processed = 0
    skipped = 0

    for cls, path in items:
        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] Nu pot citi imaginea: {path}")
            skipped += 1
            continue

        out_dir = os.path.join(OUTPUT_ROOT, cls)
        os.makedirs(out_dir, exist_ok=True)

        # 1. încercăm să detectăm roata
        bbox = detect_wheel_bbox(model, img)

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                crop = img
        else:
            # dacă nu găsește roata, presupunem că poate este deja doar tread-ul
            crop = img

        # 2. extragem pattern-ul cu extractorul nostru
        pattern = extract_tire_pattern(crop)

        if pattern is None:
            print(f"[WARN] Nu am putut extrage pattern din: {path}")
            skipped += 1
            continue

        base = os.path.basename(path)
        name, _ = os.path.splitext(base)
        out_path = os.path.join(out_dir, f"{name}_pattern.png")

        cv2.imwrite(out_path, pattern)
        processed += 1

    print("\n[INFO] Procesare terminată.")
    print(f"[INFO] Imagini procesate: {processed}")
    print(f"[INFO] Imagini sărite: {skipped}")
    print(f"[INFO] Dataset de pattern-uri la: {OUTPUT_ROOT}")


if __name__ == "__main__":
    process_dataset()
