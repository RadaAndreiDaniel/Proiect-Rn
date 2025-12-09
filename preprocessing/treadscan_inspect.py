import cv2
import numpy as np

from treadscan.detector import Detector, BackgroundSubtractorSimple, FrameExtractor
from treadscan.segmentor import Segmentor
from treadscan import unwrap, clahe, remove_gradient


def extract_treadscan(img):

    # Convert to grayscale for BackgroundSubtractorSimple
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # === Background model ===
    backsub = BackgroundSubtractorSimple(gray)

    # === Frame extractor for single images ===
    frame_extractor = FrameExtractor(
        input_path=None,
        input_type="image"
    )

    # Most versions of treadscan support calling the extractor directly
    frame = frame_extractor(img)

    # === Detector ===
    detector = Detector(backsub, frame_extractor)

    # Detect ROI
    detection = detector.detect(frame)

    if detection is None or "roi" not in detection:
        raise ValueError("Could not detect tire ROI")

    x1, y1, x2, y2 = detection["roi"]
    tire_crop = img[y1:y2, x1:x2]

    # === Segment tread ===
    segmentor = Segmentor()
    seg = segmentor(tire_crop)

    if seg is None or "mask" not in seg:
        raise ValueError("Could not segment tread mask")

    mask = seg["mask"]

    tread_only = cv2.bitwise_and(tire_crop, tire_crop, mask=mask)

    # === Unwrap ===
    unwrapped = unwrap(tread_only)

    # === Clean ===
    unwrapped = remove_gradient(unwrapped)
    unwrapped = clahe(unwrapped)

    return unwrapped


# === TEST ===
if __name__ == "__main__":

    path = r"C:\Users\DANI\OneDrive\Desktop\ProiectVideoRn\data\train\mixt\mixta_01.jpg"
    out_path = r"C:\Users\DANI\OneDrive\Desktop\ProiectVideoRn\final.jpg"

    img = cv2.imread(path)
    print("Image loaded:", img.shape)

    tread = extract_treadscan(img)

    cv2.imwrite(out_path, tread)
    print("Saved to:", out_path)
