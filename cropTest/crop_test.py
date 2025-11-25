import cv2
import numpy as np
from matplotlib import pyplot as plt

img_path = r"C:\Users\DANI\OneDrive\Desktop\ProiectRn\cropTest\test_tire2.jpg"
output_path = r"C:\Users\DANI\OneDrive\Desktop\ProiectRn\cropTest\band_pattern_highres.jpg"

img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Imaginea nu a fost gasita la {img_path}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)
edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=1)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

best_box = None
max_area = 0

for c in contours:
    rect = cv2.minAreaRect(c)
    (cx, cy), (w, h), angle = rect
    if w < h:
        w, h = h, w  # Asiguram ca w > h
    area = w * h
    ratio = w / h if h != 0 else 0
    if area > max_area and ratio > 2.5:  # raport lungime/inaltime
        max_area = area
        best_box = rect

if best_box:
    box = cv2.boxPoints(best_box)
    box = np.intp(box)
    width = int(best_box[1][0])
    height = int(best_box[1][1])
    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (width, height))
    
    # Scalare la rezolutie mare
    final_crop = cv2.resize(warped, (512, 512), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(output_path, final_crop)
    plt.imshow(cv2.cvtColor(final_crop, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    print(f"Banda de rulare a fost salvata la rezolutie mare: {output_path}")
else:
    print("Nu s-a gasit banda de rulare.")
