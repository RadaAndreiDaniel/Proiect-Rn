from ultralytics import YOLO
import cv2
import os

def detect_wheel_and_crop(frame_path, output_dir, model_path="yolov8n.pt"):
    """
    Detectează roata folosind YOLO și o decupează complet într-un PNG.
    """

    # Creează directorul de output dacă nu există
    os.makedirs(output_dir, exist_ok=True)

    # Încarcă modelul YOLO
    model = YOLO(model_path)

    # Încarcă imaginea
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"Eroare: nu pot încărca imaginea {frame_path}")
        return None

    # Rulăm detectarea
    results = model(frame)

    # Căutăm clasa "tire" sau "wheel"
    best_box = None

    for result in results:
        for box in result.boxes:
            cls = int(box.cls)
            conf = float(box.conf)

            # YOLO COCO: wheel / tire apare ca parte din "car" sau "truck", deci luăm bounding box-ul cu scor mare
            if conf > 0.40:
                best_box = box
                break

    if best_box is None:
        print(f"[!] Nu s-a detectat roata în {frame_path}")
        return None

    # Extragem coordonatele
    x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())

    # Crop roata completă
    wheel_crop = frame[y1:y2, x1:x2]

    # Salvăm
    out_name = os.path.basename(frame_path).replace(".png", "_wheel.png")
    out_path = os.path.join(output_dir, out_name)
    cv2.imwrite(out_path, wheel_crop)

    return out_path


if __name__ == "__main__":

    input_folder = "data/processed/frames_raw/"
    output_folder = "data/processed/wheel_crops/"

    for img in os.listdir(input_folder):
        if img.endswith(".png"):
            detect_wheel_and_crop(
                os.path.join(input_folder, img),
                output_folder
            )
