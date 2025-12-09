import cv2
import os

def extract_frames(video_path, output_dir):
    """
    Extrage TOATE cadrele dintr-un fișier video și le salvează ca PNG,
    cu nume bazat pe timestamp-ul în milisecunde.
    """

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video-ul nu există: {video_path}")

    # Creează directorul de output dacă nu există
    os.makedirs(output_dir, exist_ok=True)

    # Deschide video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Nu pot deschide video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1:
        print("[AVERTISMENT] FPS necunoscut sau invalid. Folosesc fallback: 30 FPS.")
        fps = 30

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] FPS: {fps}")
    print(f"[INFO] Total cadre în video: {frame_count}")
    print(f"[INFO] Încep extragerea cadrelor...")

    count = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Final de video

        # Timpul în milisecunde al cadrului curent
        timestamp_ms = int((count / fps) * 1000)

        # Nume fișier
        out_name = f"{timestamp_ms:08d}ms.png"
        out_path = os.path.join(output_dir, out_name)

        # Salvează cadrul ca PNG
        cv2.imwrite(out_path, frame)
        saved += 1

        count += 1

    cap.release()
    print(f"[FINALIZAT] Cadre extrase: {saved}")
    return saved


if __name__ == "__main__":
    # EXEMPLE DE UTILIZARE (poți modifica după nevoie):

    video_input = "data/raw/masina1.mp4"
    output_frames = "data/processed/frames_raw/"

    extract_frames(video_input, output_frames)
