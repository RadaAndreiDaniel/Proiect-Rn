import os
import cv2
import numpy as np
import treadscan


# ====== CONFIG ======
MODEL_PATH = r"C:\Users\DANI\OneDrive\Desktop\ProiectVideoRn\RCNN_model\saved_model.pth"
IMAGE_PATH = r"C:\Users\DANI\OneDrive\Desktop\ProiectVideoRn\data\train\mixt\mixta_01.jpg"

OUT_TREAD = r"C:\Users\DANI\OneDrive\Desktop\ProiectVideoRn\final_tread.jpg"
OUT_DEBUG = r"C:\Users\DANI\OneDrive\Desktop\ProiectVideoRn\final_debug.jpg"


def extract_tread_from_image_rcnn(bgr_img,
                                  model_path,
                                  confidence_threshold=0.7,
                                  output_size=(256, 512),
                                  debug=False):
    """
    Extrage banda de rulare dintr-o imagine folosind:
      - SegmentorRCNN (Keypoint RCNN)
      - TireModel.unwrap

    bgr_img: imagine BGR (cv2.imread)
    model_path: calea la RCNN_model/saved_model.pth
    return: (tread_img, debug_img_optional)
    """

    # 1. Convertim la grayscale (RCNN in treadscan lucreaza pe grayscale)
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    # 2. Incarcam modelul RCNN
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"RCNN model not found at: {model_path}")

    segmentor = treadscan.SegmentorRCNN(model_path, use_cuda=False)

    # 3. Cauta toate anvelopele in imagine
    keypoints_list = segmentor.find_keypoints(
        gray,
        confidence_threshold=confidence_threshold,
        iou_threshold=0.1
    )

    if not keypoints_list:
        raise RuntimeError("No tires found by SegmentorRCNN in this image.")

    # Pentru moment luam prima anvelopa gasita
    keypoints = keypoints_list[0]

    # 4. Construim modelul de anvelopa si "desfasuram" banda de rulare
    h, w = gray.shape
    tire_model = treadscan.TireModel((h, w))
    tire_model.from_keypoints(*keypoints)

    # unwrap() foloseste unghiuri default: start=-10, end=80
    tread = tire_model.unwrap(gray)

    # 5. Postprocesare recomandata in README:
    #    remove_gradient + clahe
    tread = treadscan.remove_gradient(tread)
    tread = treadscan.clahe(tread)

    # Normalizam si redimensionam la o marime fixa (pt. CNN)
    tread = cv2.normalize(tread, None, 0, 255, cv2.NORM_MINMAX)
    tread = tread.astype("uint8")
    tread = cv2.resize(tread, output_size, interpolation=cv2.INTER_AREA)

    tread_bgr = cv2.cvtColor(tread, cv2.COLOR_GRAY2BGR)

    debug_img = None
    if debug:
        # Desenam modelul anvelopei peste original, ca sa vezi ce a detectat
        debug_img = bgr_img.copy()
        debug_img = tire_model.draw(debug_img, color=(0, 255, 0), thickness=2)

    return tread_bgr, debug_img


if __name__ == "__main__":
    print("[INFO] Loading image:", IMAGE_PATH)
    img = cv2.imread(IMAGE_PATH)

    if img is None:
        raise RuntimeError(f"Could not read image: {IMAGE_PATH}")

    tread_img, debug_img = extract_tread_from_image_rcnn(
        img,
        MODEL_PATH,
        confidence_threshold=0.7,
        output_size=(256, 512),
        debug=True
    )

    # Salvam rezultatele
    os.makedirs(os.path.dirname(OUT_TREAD), exist_ok=True)
    cv2.imwrite(OUT_TREAD, tread_img)
    print("[INFO] Saved tread to:", OUT_TREAD)

    if debug_img is not None:
        cv2.imwrite(OUT_DEBUG, debug_img)
        print("[INFO] Saved debug image to:", OUT_DEBUG)

    print("[DONE]")
