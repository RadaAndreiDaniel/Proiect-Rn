import cv2
from extract_tire_pattern import extract_tire_pattern
from PIL import Image
import numpy as np

# loader universal (citeste .jpg, .png, .avif etc.)
def load_image_any(path):
    pil_img = Image.open(path).convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# === pune aici calea imaginii tale ===
img_path = r"C:\Users\DANI\OneDrive\Desktop\ProiectVideoRn\data\train\vara\varatest.webp"

img = load_image_any(img_path)

print("Imaginea a fost incarcata:", img.shape)

pattern = extract_tire_pattern(img)

output_path = "rezultat_corect.png"
cv2.imwrite(output_path, pattern)

print("Pattern extras salvat in:", output_path)
