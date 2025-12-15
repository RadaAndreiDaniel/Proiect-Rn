import cv2
import numpy as np
import os
from PIL import Image

# --- 1. Citire AVIF sau orice format ---
def load_image(path):
    ext = path.lower().split(".")[-1]
    if ext in ["avif", "heic", "heif"]:
        # folosim PIL pentru AVIF
        pil_img = Image.open(path).convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    else:
        return cv2.imread(path)


# --- 2. Extrage strict zona cu pattern ---
def extrage_pattern(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Contrast local puternic
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # Noise filtering
    blur = cv2.GaussianBlur(gray, (3,3), 0)

    # Edge detection adaptiv
    v = np.median(blur)
    lower = int(max(0, 0.66*v))
    upper = int(min(255, 1.33*v))
    edges = cv2.Canny(blur, lower, upper)

    # Morfologie pentru a întări liniile
    kernel = np.ones((5,5), np.uint8)
    morphed = cv2.dilate(edges, kernel, iterations=2)
    morphed = cv2.erode(morphed, kernel, iterations=1)

    # Găsim cel mai mare obiect = pattern-ul benzii
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return gray   # fallback

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # Crop exact pe zona de pattern
    crop = gray[y:y+h, x:x+w]

    # Resize uniform
    crop = cv2.resize(crop, (256, 256))

    # Contrast final
    crop = cv2.equalizeHist(crop)

    return crop


# --- 3. Procesare automată pe foldere ---
input_root = r"C:\Users\DANI\OneDrive\Desktop\Proiect_Rada_Andrei_Daniel_Rn\data\raw"          # aici sunt folderele tale vara/iarna/mixt
output_root = r"C:\Users\DANI\OneDrive\Desktop\Proiect_Rada_Andrei_Daniel_Rn\data\processed"    # aici vor fi salvate imaginile filtrate

os.makedirs(output_root, exist_ok=True)

folders = ["vara", "iarna", "mixt"]

for folder in folders:
    in_path = os.path.join(input_root, folder)
    out_path = os.path.join(output_root, folder)
    os.makedirs(out_path, exist_ok=True)

    for file in os.listdir(in_path):
        file_path = os.path.join(in_path, file)

        # ignorăm fișiere non-imagine
        if not file.lower().endswith(("jpg","jpeg","png","bmp","avif","webp")):
            continue

        print("Procesez:", file_path)

        img = load_image(file_path)
        if img is None:
            print("❌ NU pot citi:", file_path)
            continue

        rezultat = extrage_pattern(img)

        # salvare rezultat
        out_file = os.path.join(out_path, file.rsplit(".",1)[0] + "_proc.jpg")
        cv2.imwrite(out_file, rezultat)

print("\n✔ GATA! Toate imaginile au fost procesate în folderul /processed\n")
