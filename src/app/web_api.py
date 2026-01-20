import sys
from pathlib import Path
import traceback
from flask import Flask, request, jsonify, send_from_directory, render_template

CURRENT_FILE = Path(__file__).resolve()
APP_PATH = CURRENT_FILE.parent
SRC_PATH = APP_PATH.parent
ROOT_PATH = SRC_PATH.parent

if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from finalApp2 import (
    metoda_yolo_image_return_with_bbox,
    metoda_manual_image_return,
    metoda_yolo_video_return
)

app = Flask(__name__, template_folder="templates")

UPLOADS = APP_PATH / "uploads"
UPLOADS.mkdir(exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["file"]
        method = request.form.get("method", "yolo")

        path = UPLOADS / file.filename
        file.save(path)

        suffix = path.suffix.lower()

        # VIDEO
        if suffix in [".mp4", ".avi", ".mov", ".mkv"]:
            label, probs, bbox_path, pattern_path = \
                metoda_yolo_video_return(path)

        # MANUAL IMAGINE
        elif method == "manual":
            label, probs, bbox_path, pattern_path = \
                metoda_manual_image_return(path)

        # YOLO IMAGINE
        else:
            label, probs, bbox_path, pattern_path = \
                metoda_yolo_image_return_with_bbox(path)

        return jsonify({
            "status": "ok",
            "label": label,
            "probabilities": probs,
            "bbox_url": f"/result/{bbox_path.name}",
            "crop_url": f"/result/{pattern_path.name}"
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/result/<filename>")
def result_image(filename):
    return send_from_directory(ROOT_PATH, filename)

# 🔑 ENDPOINT PENTRU VIDEO PREVIEW
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOADS, filename)

if __name__ == "__main__":
    app.run(debug=True)
