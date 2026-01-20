# =========================
# PATH SETUP
# =========================
import sys
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
APP_PATH = CURRENT_FILE.parent
SRC_PATH = APP_PATH.parent
ROOT_PATH = SRC_PATH.parent

if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

# =========================
# IMPORTS
# =========================
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO
from data_acquisition.annotator import Annotator

# =========================
# CONFIG
# =========================
YOLO_MODEL_PATH = ROOT_PATH / "models" / "best.pt"

# IMPORTANT: folosim checkpoint (contine best model + class_names)
FINAL_MODEL_PATH = ROOT_PATH / "models" / "trained_model.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 224
PATTERN_SIZE = 256

# class_names vor veni din checkpoint (ca sa nu existe mismatch)
CLASS_NAMES = None

# =========================
# UTIL
# =========================
def resize_for_yolo(img, max_width=1000):
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    scale = max_width / w
    return cv2.resize(img, (max_width, int(h * scale)))

def get_blur_score(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

# =========================
# YOLO (CACHED)
# =========================
_YOLO = None

def load_yolo_once():
    global _YOLO
    if _YOLO is None:
        _YOLO = YOLO(str(YOLO_MODEL_PATH))
    return _YOLO

# =========================
# CLASIFICATOR (CHECKPOINT, NORMALIZE, CACHED)
# =========================
_MODEL = None
_CLASS_NAMES = None

def load_final_model_once():
    """
    Incarca o singura data modelul + ordinea claselor din resnet18_checkpoint.pt
    ca sa ai aceleasi rezultate ca in scriptul separat.
    """
    global _MODEL, _CLASS_NAMES
    if _MODEL is not None and _CLASS_NAMES is not None:
        return _MODEL, _CLASS_NAMES

    ckpt = torch.load(FINAL_MODEL_PATH, map_location=DEVICE)
    _CLASS_NAMES = ckpt["class_names"]  # ex: ['iarna','mixt','vara']

    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(_CLASS_NAMES))
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()

    _MODEL = model
    return _MODEL, _CLASS_NAMES

def preprocess_for_classifier(image_path):
    """
    IMPORTANT: exact ca la training/eval:
    Resize -> ToTensor -> Normalize(ImageNet)
    """
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    img = Image.open(image_path).convert("RGB")
    return tf(img).unsqueeze(0).to(DEVICE)

def classify_pattern_return(image_path):
    model, class_names = load_final_model_once()
    x = preprocess_for_classifier(image_path)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]  # tensor [3]

    idx = int(torch.argmax(probs).item())

    return class_names[idx], {
        cls: float(p.item() * 100.0)
        for cls, p in zip(class_names, probs)
    }

# =========================
# NORMALIZE (INPUT NN)
# =========================
def normalize_pattern(img):
    # asigurare grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = img.shape
    mx, my = int(0.15 * w), int(0.15 * h)

    img = img[my:h-my, mx:w-mx]
    img = cv2.equalizeHist(img)
    img = cv2.resize(img, (PATTERN_SIZE, PATTERN_SIZE))
    return img

# =========================
# YOLO – IMAGINE
# =========================
def metoda_yolo_image_return_with_bbox(image_path):
    model = load_yolo_once()

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError("Imagine invalidă")

    img_small = resize_for_yolo(img)

    results = model(img_small)
    if not results or not results[0].boxes:
        raise ValueError("YOLO nu a detectat nimic")

    box = results[0].boxes[0]
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    boxed = img_small.copy()
    cv2.rectangle(boxed, (x1, y1), (x2, y2), (0, 255, 0), 3)

    crop = img_small[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("Crop invalid")

    pattern = normalize_pattern(crop)

    bbox_path = ROOT_PATH / "boxed_image.jpg"
    pattern_path = ROOT_PATH / "pattern_from_image.jpg"

    cv2.imwrite(str(bbox_path), boxed)
    cv2.imwrite(str(pattern_path), pattern)

    label, probs = classify_pattern_return(pattern_path)
    return label, probs, bbox_path, pattern_path

# =========================
# MANUAL – IMAGINE
# =========================
def metoda_manual_image_return(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError("Imagine invalidă")

    img_for_annotator = resize_for_yolo(img, max_width=1200)
    gray = cv2.cvtColor(img_for_annotator, cv2.COLOR_BGR2GRAY)

    annotator = Annotator(gray, max_width=1200, max_height=800)
    if not annotator.annotate_keypoints():
        raise ValueError("Anulare manuală")

    preview = annotator._last_preview
    pattern = normalize_pattern(preview)

    bbox_path = ROOT_PATH / "boxed_manual.jpg"
    pattern_path = ROOT_PATH / "pattern_from_manual.jpg"

    cv2.imwrite(str(bbox_path), preview)
    cv2.imwrite(str(pattern_path), pattern)

    label, probs = classify_pattern_return(pattern_path)
    return label, probs, bbox_path, pattern_path

# =========================
# VIDEO – CADRU OPTIM + CLARITATE
# =========================
def metoda_yolo_video_return(video_path):
    model = load_yolo_once()
    cap = cv2.VideoCapture(str(video_path))

    best_crop = None
    best_frame = None
    best_box = None
    max_clarity = 0
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % 3 != 0:
            continue

        results = model(frame)
        if not results or not results[0].boxes:
            continue

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            clarity = get_blur_score(crop)
            if clarity > max_clarity:
                max_clarity = clarity
                best_crop = crop.copy()
                best_frame = frame.copy()
                best_box = (x1, y1, x2, y2)

    cap.release()

    if best_crop is None:
        raise ValueError("Nu s-a găsit cadru valid în video")

    x1, y1, x2, y2 = best_box
    boxed = best_frame.copy()
    cv2.rectangle(boxed, (x1, y1), (x2, y2), (0, 255, 0), 3)

    pattern = normalize_pattern(best_crop)

    bbox_path = ROOT_PATH / "boxed_video.jpg"
    pattern_path = ROOT_PATH / "pattern_from_video.jpg"

    cv2.imwrite(str(bbox_path), resize_for_yolo(boxed))
    cv2.imwrite(str(pattern_path), pattern)

    label, probs = classify_pattern_return(pattern_path)
    return label, probs, bbox_path, pattern_path
