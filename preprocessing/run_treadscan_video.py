import cv2
import os
import treadscan

# ---- CONFIG ----
VIDEO_PATH = r"C:\Users\DANI\OneDrive\Desktop\ProiectVideoRn\data\test\roata2.mp4"
OUTPUT_DIR = r"C:\Users\DANI\OneDrive\Desktop\ProiectVideoRn\output"
MODEL_PATH = r"C:\Users\DANI\OneDrive\Desktop\ProiectVideoRn\models\saved_model.pth"
BACKGROUND_IMAGE = None     # Dacă ai nevoie de background subtraction

# Creează folderul output dacă nu există
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----- 1. Încarcă video-ul -----
frame_extractor = treadscan.FrameExtractor(VIDEO_PATH, treadscan.InputType.VIDEO)

# ----- 2. Segmentor cu RCNN -----
segmentor = treadscan.SegmentorRCNN(MODEL_PATH)

# ----- 3. Detectare + analiză cadre (fără background subtractor) -----
detector = treadscan.Detector(backsub=None, frame_extractor=frame_extractor)

print("Procesare video...")

i = 1
for image in detector.detect():

    # salvăm cadrul original
    frame_path = os.path.join(OUTPUT_DIR, f"frame_{i:04d}.jpg")
    cv2.imwrite(frame_path, image)

    # detectăm punctele cheie
    keypoints_list = segmentor.find_keypoints(image)

    for j, keypoints in enumerate(keypoints_list):

        # reconstruim modelul anvelopei
        tire_model = treadscan.TireModel(image.shape)
        tire_model.from_keypoints(*keypoints)

        # unwrap = extragerea modelului benzii de rulare
        tread = tire_model.unwrap(image)

        # post-procesare
        tread = treadscan.remove_gradient(tread)
        tread = treadscan.clahe(tread)

        tread_path = os.path.join(OUTPUT_DIR, f"tread_{i:04d}_{j:02d}.jpg")
        cv2.imwrite(tread_path, tread)

        print(f"✔ Tread extras: {tread_path}")

    i += 1

print("Procesare finalizata!")
