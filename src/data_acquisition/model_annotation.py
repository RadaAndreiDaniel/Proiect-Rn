"""
Model annotation script
-----------------------
✔ Annotates tires using GUI
✔ Saves annotated image
✔ Saves JSON with keypoints
✔ Saves tread preview (the image shown in top-right corner) resized to 256×256 px
"""

from argparse import ArgumentParser
from pathlib import Path
import cv2
from annotator import Annotator


def main(filename: str, max_width: int, max_height: int, rescale: float):

    path = Path(filename)

    # Determine input files
    if path.is_dir():
        filenames = [str(f) for f in path.iterdir() if f.is_file()]
        filenames.sort()
    else:
        filenames = [filename]

    for file in filenames:

        # Load image (as grayscale)
        img = cv2.imread(file)
        if img is None:
            print("Error: Cannot read file:", file)
            continue

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        annotator = Annotator(img, max_width, max_height)
        result = annotator.annotate_keypoints()

        if not result:
            print("[WARN] No annotation result for:", file)
            continue

        parent = Path(file).parent
        stem = Path(file).stem

        # -------------------------------
        # OUTPUT FOLDERS
        # -------------------------------
        images_folder = parent / "images_"
        annotations_folder = parent / "annotations_"

        images_folder.mkdir(exist_ok=True)
        annotations_folder.mkdir(exist_ok=True)


        # -------------------------------
        # SAVE ANNOTATED IMAGE
        # -------------------------------
        annotated_path = images_folder / f"{stem}_annotated.jpg"
        cv2.imwrite(str(annotated_path), annotator.image)
        print("[OK] Saved annotated image:", annotated_path)


        # -------------------------------
        # SAVE JSON
        # -------------------------------
        json_path = annotations_folder / f"{stem}.json"
        with open(json_path, "w") as f:
            f.write(result)
        print("[OK] Saved JSON:", json_path)


        # -------------------------------
        # SAVE PREVIEW (RESIZED TO 256×256)
        # -------------------------------
        if hasattr(annotator, "_last_preview") and annotator._last_preview is not None:

            preview_resized = cv2.resize(
                annotator._last_preview,
                (256, 256),
                interpolation=cv2.INTER_AREA
            )

            preview_path = images_folder / f"{stem}_preview_256.jpg"
            cv2.imwrite(str(preview_path), preview_resized)

            print("[OK] Saved preview at 256×256 px:", preview_path)
        else:
            print("[WARN] No preview available to save.")



# ---------------------------------
# CLI SETUP
# ---------------------------------
parser = ArgumentParser(description="Annotate tire tread manually and save results.")
parser.add_argument("-i", dest="filename", required=True, help="input image or folder")
parser.add_argument("--width", type=int, default=1800, help="max GUI width")
parser.add_argument("--height", type=int, default=800, help="max GUI height")
parser.add_argument("--rescale", type=float, default=1.0, help="(unused)")

args = parser.parse_args()

if __name__ == "__main__":
    main(args.filename, args.width, args.height, args.rescale)
