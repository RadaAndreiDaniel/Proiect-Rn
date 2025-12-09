import cv2
import numpy as np

def extract_tire_pattern(img, output_size=(256, 512)):
    """
    Universal tire tread extractor.
    - Handles top-down images (tread seen from above).
    - Handles oblique / side images, tread left or right.
    - Avoids rim by combining texture and brightness.
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # ============================
    # 0. TOP-DOWN DETECTION
    # ============================
    # imagine inalta si ingusta -> cel mai probabil vazuta de sus
    aspect_ratio = h / max(w, 1)

    if aspect_ratio > 1.3:
        print("Detected TOP-DOWN view")

        # luam 50% din centrul imaginii (tread-ul central)
        x1 = int(w * 0.25)
        x2 = int(w * 0.75)
        crop = gray[:, x1:x2]

        crop = cv2.equalizeHist(crop)
        crop = cv2.resize(crop, output_size, interpolation=cv2.INTER_AREA)

        return cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)

    # ============================
    # 1. OBLIQUE / ANY-ANGLE VIEW
    # ============================
    print("Detected OBLIQUE / ANGLED view")

    # 1.1 Usor blur ca sa reducem zgomotul
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 1.2 Textura verticala (striatii) cu Sobel X
    sobelx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    texture = np.abs(sobelx)

    # 1.3 Profil pe coloane:
    #    - col_tex: cat de multa textura verticala
    #    - col_int: cat de luminoasa e coloana (janta e foarte luminoasa)
    col_tex = texture.mean(axis=0)
    col_int = gray.mean(axis=0)

    # Normalizam
    col_tex_norm = col_tex / (col_tex.max() + 1e-6)
    col_int_norm = col_int / 255.0

    # 1.4 Scor pentru tread:
    #    vrem textura mare, luminozitate mai degraba medie/intunecata
    score = col_tex_norm * (1.0 - col_int_norm)

    # netezim scorul pe orizontala ca sa nu fie rupt in bucati mici
    score = cv2.GaussianBlur(score.reshape(1, -1), (1, 9), 0).flatten()

    # 1.5 Threshold relativ: luam coloanele cu scor suficient de mare
    thresh = 0.5 * score.max()
    mask_cols = score > thresh
    xs = np.where(mask_cols)[0]

    if len(xs) == 0:
        # fallback: nu am gasit nimic sigur, returnam imaginea redimensionata
        print("Warning: could not detect tread region, fallback to full image.")
        fallback = cv2.equalizeHist(gray)
        fallback = cv2.resize(fallback, output_size, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(fallback, cv2.COLOR_GRAY2BGR)

    x_min, x_max = xs[0], xs[-1]

    # daca zona este foarte ingusta, o mai largim putin
    min_width = int(0.08 * w)  # cel putin 8% din latime
    if (x_max - x_min) < min_width:
        center = (x_min + x_max) // 2
        x_min = max(center - min_width // 2, 0)
        x_max = min(center + min_width // 2, w - 1)

    # 1.6 Extragem banda de rulare
    tread = gray[:, x_min:x_max]

    # 1.7 Taiem putin sus/jos (5%) ca sa scapam de margini
    top = int(0.05 * h)
    bottom = int(0.95 * h)
    tread = tread[top:bottom, :]

    # 1.8 Enhancements: CLAHE + sharpening + gamma
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    tread = clahe.apply(tread)

    kernel = np.array([[0, -0.5, 0],
                       [-0.5, 3.0, -0.5],
                       [0, -0.5, 0]])
    tread = cv2.filter2D(tread, -1, kernel)

    gamma = 0.85
    tread = np.power(tread / 255.0, gamma)
    tread = (tread * 255).astype(np.uint8)

    # 1.9 Resize final
    tread = cv2.resize(tread, output_size, interpolation=cv2.INTER_AREA)

    return cv2.cvtColor(tread, cv2.COLOR_GRAY2BGR)
