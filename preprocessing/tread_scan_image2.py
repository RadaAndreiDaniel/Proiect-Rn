import cv2
import treadscan
import torch

MODEL_PATH = r"C:\Users\DANI\OneDrive\Desktop\ProiectVideoRn\RCNN_model\saved_model.pth"
IMAGE_PATH = r"C:\Users\DANI\OneDrive\Desktop\ProiectVideoRn\data\train\mixt\mixta_01.jpg"

def load_rcnn_model(path):
    print("[INFO] Loading RCNN model...")
    try:
        # PyTorch 2.6+ requires weights_only=False for legacy models
        state_dict = torch.load(path, map_location="cpu", weights_only=False)
        return state_dict
    except Exception as e:
        print("[ERR] Cannot load model:", e)
        raise

def extract_tread_from_image(image):
    # Load RCNN segmentor
    seg = treadscan.SegmentorRCNN(MODEL_PATH, use_cuda=False)

    # Detect keypoints
    keypoints_list = seg.find_keypoints(image)
    if not keypoints_list:
        print("[ERR] No tire detected.")
        return None

    # Use the first detected tire
    keypoints = keypoints_list[0]

    # Build tire model
    tire_model = treadscan.TireModel(image.shape)
    tire_model.from_keypoints(*keypoints)

    # Unwrap tread
    tread = tire_model.unwrap(image)

    # Post-processing
    tread = treadscan.remove_gradient(tread)
    tread = treadscan.clahe(tread)

    return tread


if __name__ == "__main__":
    print("[INFO] Loading image:", IMAGE_PATH)
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise RuntimeError("Could not load image.")

    tread = extract_tread_from_image(img)

    if tread is not None:
        cv2.imwrite("final_tread.jpg", tread)
        print("[DONE] Saved tread to final_tread.jpg")
    else:
        print("[FAIL] Could not extract tread.")
