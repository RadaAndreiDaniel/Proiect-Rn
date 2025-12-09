from treadscan.extractor import extract_tire
import cv2

image_path = r"C:\Users\DANI\OneDrive\Desktop\ProiectVideoRn\data\train\iarna\winter9.jpg"

img = cv2.imread(image_path)
if img is None:
    raise ValueError("Could not load image")

print("Image loaded:", img.shape)

# Extract tread using the main function
tread = extract_tire(img)

out_path = r"C:\Users\DANI\Desktop\tread_output.png"
cv2.imwrite(out_path, tread)

print("Tread extracted successfully:", out_path)
