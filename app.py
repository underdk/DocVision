import os
import io
import json

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    jsonify,
)
from PIL import Image, ImageDraw, ImageFont
import qrcode

# -----------------------------
# YOLO INIT
# -----------------------------
try:
    from ultralytics import YOLO  # type: ignore
    import numpy as np  # type: ignore

    HAS_NUMPY = True
    MODEL_PATH = os.path.join("model", "best.pt")
    if os.path.exists(MODEL_PATH):
        yolo_model = YOLO(MODEL_PATH)
    else:
        yolo_model = None
except Exception:
    yolo_model = None
    HAS_NUMPY = False

# -----------------------------
# OpenCV INIT
# -----------------------------
try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    HAS_CV2 = True
except ImportError:
    cv2 = None  # type: ignore
    HAS_CV2 = False

# -----------------------------
# FLASK INIT
# -----------------------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.secret_key = "dev-secret-change-this"  # for session

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs("model", exist_ok=True)

# Where we store all annotations (for all files) in one JSON, like your example
ANNOTATIONS_PATH = "annotations.json"


# -----------------------------
# OPTIONAL HELPERS (QR/signature/stamp generators)
# -----------------------------
def generate_sample_signature(size=(200, 50)):
    img = Image.new("RGBA", size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except IOError:
        font = ImageFont.load_default()
    text = "Signature"
    bbox = font.getbbox(text)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    position = ((size[0] - tw) // 2, (size[1] - th) // 2)
    draw.text(position, text, fill="black", font=font)
    return img


def generate_sample_stamp(size=(150, 150)):
    img = Image.new("RGBA", size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    radius = min(size) // 2 - 5
    center = (size[0] // 2, size[1] // 2)
    draw.ellipse(
        [
            center[0] - radius,
            center[1] - radius,
            center[0] + radius,
            center[1] + radius,
        ],
        outline="red",
        width=4,
    )
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except IOError:
        font = ImageFont.load_default()
    text = "STAMP"
    bbox = font.getbbox(text)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pos = (center[0] - tw // 2, center[1] - th // 2)
    draw.text(pos, text, fill="red", font=font)
    return img


def generate_qr_code(data="Sample QR", size=200):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)
    img_qr = qr.make_image(fill_color="black", back_color="white").convert("RGBA")
    img_qr = img_qr.resize((size, size))
    return img_qr


# -----------------------------
# DETECTION BACKENDS
# -----------------------------
def detect_elements_cv2(pil_img):
    """
    Fallback: simple OpenCV heuristics for qr / stamp / signature.
    Returns: list of (label, (x1, y1, x2, y2)).
    """
    if not HAS_CV2 or cv2 is None or np is None:
        return []

    img = np.array(pil_img.convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    height, width = img_bgr.shape[:2]
    detections = []

    # ---- QR codes ----
    try:
        qr_detector = cv2.QRCodeDetector()
        retval, decoded_info, points, _ = qr_detector.detectAndDecodeMulti(img_bgr)
        if retval and points is not None:
            for quad in points:
                quad = quad.astype(int)
                x1 = max(0, int(np.min(quad[:, 0])))
                y1 = max(0, int(np.min(quad[:, 1])))
                x2 = min(width - 1, int(np.max(quad[:, 0])))
                y2 = min(height - 1, int(np.max(quad[:, 1])))
                detections.append(("qr", (x1, y1, x2, y2)))
    except Exception:
        pass

    # ---- Stamps (red blobs) ----
    try:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(
            red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1000:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / float(h)
            if 0.5 < aspect < 2.0:
                detections.append(("stamp", (x, y, x + w, y + h)))
    except Exception:
        pass

    # ---- Signatures (long thin dark shapes) ----
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            15,
            8,
        )
        kernel = np.ones((3, 3), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(
            morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500 or area > 20000:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            if aspect_ratio > 2.5 and h < height * 0.2:
                detections.append(("signature", (x, y, x + w, y + h)))
    except Exception:
        pass

    return detections


def detect_elements(pil_img):
    """
    Use YOLO if available, otherwise OpenCV fallback.
    Returns: list of (label, (x1, y1, x2, y2)).
    """
    if yolo_model is not None and HAS_NUMPY:
        try:
            img_np = np.array(pil_img.convert("RGB"))
            results = yolo_model.predict(img_np, verbose=False)
            detections = []
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls.item())
                    cls_name = r.names.get(cls_id, "")
                    label = (
                        cls_name.lower()
                        .replace("qr-code", "qr")
                        .replace("qrcode", "qr")
                        .replace("qr code", "qr")
                    )
                    if label not in ["qr", "signature", "stamp"]:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    width, height = pil_img.size
                    x1 = max(0, min(int(x1), width - 1))
                    y1 = max(0, min(int(y1), height - 1))
                    x2 = max(0, min(int(x2), width - 1))
                    y2 = max(0, min(int(y2), height - 1))
                    detections.append((label, (x1, y1, x2, y2)))
            return detections
        except Exception:
            pass

    return detect_elements_cv2(pil_img)


# -----------------------------
# JSON ANNOTATIONS HELPERS
# -----------------------------
def load_annotations_master():
    """Load the global annotations.json (if exists)."""
    if os.path.exists(ANNOTATIONS_PATH):
        try:
            with open(ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_annotations_for_file(filename, file_annotations):
    """
    Update annotations.json so that:
    {
      "file1.pdf": { "page_1": {...}, ... },
      "file2.pdf": { ... }
    }
    in the same general format as your example JSON files.
    """
    master = load_annotations_master()
    master[filename] = file_annotations
    with open(ANNOTATIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(master, f, ensure_ascii=False, indent=2)


# -----------------------------
# ROUTES
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    # multi-upload
    if request.method == "POST":
        if "files" in request.files:
            files_uploaded = request.files.getlist("files")
            for f in files_uploaded:
                if not f or f.filename == "":
                    continue
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], f.filename)
                f.save(filepath)
        return redirect(url_for("index"))

    files = sorted(os.listdir(app.config["UPLOAD_FOLDER"]))
    return render_template("index.html", files=files)


@app.route("/delete/<filename>", methods=["POST"])
def delete_file(filename):
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if os.path.exists(filepath):
        os.remove(filepath)

    # also remove from annotations.json if present
    if os.path.exists(ANNOTATIONS_PATH):
        master = load_annotations_master()
        if filename in master:
            del master[filename]
            with open(ANNOTATIONS_PATH, "w", encoding="utf-8") as f:
                json.dump(master, f, ensure_ascii=False, indent=2)

    return redirect(url_for("index"))


@app.route("/annotations/<filename>")
def annotations_for_file(filename):
    """
    Optional: show JSON annotations for a given file
    in browser: /annotations/<filename>.
    """
    master = load_annotations_master()
    file_data = master.get(filename)
    if file_data is None:
        return jsonify({"error": "No annotations for this file"}), 404
    return jsonify({filename: file_data})


@app.route("/process/<filename>")
def process_image(filename):
    """
    Open a file and run detection with global settings.
    Detection results are:
      1) drawn on preview images;
      2) saved into annotations.json in a structure
         similar to selected_annotations.json/masked_annotations.json.
    """
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if not os.path.exists(filepath):
        return redirect(url_for("index"))

    files = sorted(os.listdir(app.config["UPLOAD_FOLDER"]))
    prev_file = next_file = None
    if filename in files:
        idx = files.index(filename)
        if idx > 0:
            prev_file = files[idx - 1]
        if idx < len(files) - 1:
            next_file = files[idx + 1]

    # ---- read detection preferences from URL or session ----
    def latest_param(name, session_key, default_value=True):
        vals = request.args.getlist(name)
        if vals:
            # last wins (can be "0" or "1")
            return vals[-1] == "1"
        return session.get(session_key, default_value)

    highlight_qr = latest_param("qr", "highlight_qr", True)
    highlight_sig = latest_param("sig", "highlight_sig", True)
    highlight_stamp = latest_param("stamp", "highlight_stamp", True)

    session["highlight_qr"] = highlight_qr
    session["highlight_sig"] = highlight_sig
    session["highlight_stamp"] = highlight_stamp

    import base64

    ext = os.path.splitext(filename)[1].lower()
    images_base64 = []
    file_annotations = {}  # will be saved into annotations.json
    annotation_counter = 0

    # -----------------------------
    # PDF
    # -----------------------------
    if ext == ".pdf":
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(filepath)
            page_count = doc.page_count

            for page_index in range(page_count):
                page = doc.load_page(page_index)
                pix = page.get_pixmap()
                img_bytes = pix.tobytes("png")
                pil_page = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
                width, height = pil_page.size

                detections = detect_elements(pil_page)
                draw = ImageDraw.Draw(pil_page)
                colors = {
                    "qr": (0, 128, 255, 255),
                    "signature": (0, 200, 0, 255),
                    "stamp": (255, 0, 0, 255),
                }

                page_ann_list = []
                for label, (x1, y1, x2, y2) in detections:
                    # apply filters (what user chose)
                    if label == "qr" and not highlight_qr:
                        continue
                    if label == "signature" and not highlight_sig:
                        continue
                    if label == "stamp" and not highlight_stamp:
                        continue

                    color = colors.get(label, (255, 255, 0, 255))
                    draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=4)
                    text = label.upper()
                    try:
                        font = ImageFont.truetype("arial.ttf", 20)
                    except IOError:
                        font = ImageFont.load_default()
                    bbox_text = font.getbbox(text)
                    tw, th = bbox_text[2] - bbox_text[0], bbox_text[3] - bbox_text[1]
                    text_bg = (x1, y1 - th - 4, x1 + tw + 4, y1)
                    bg_color = (*color[:3], 200)
                    draw.rectangle(text_bg, fill=bg_color)
                    draw.text(
                        (x1 + 2, y1 - th - 2),
                        text,
                        fill=(255, 255, 255, 255),
                        font=font,
                    )

                    # ---- build JSON annotation for this detection ----
                    annotation_counter += 1
                    w_box = x2 - x1
                    h_box = y2 - y1
                    ann_entry = {
                        f"annotation_{annotation_counter}": {
                            "category": label,
                            "bbox": {
                                "x": float(x1),
                                "y": float(y1),
                                "width": float(w_box),
                                "height": float(h_box),
                            },
                            "area": float(w_box * h_box),
                        }
                    }
                    page_ann_list.append(ann_entry)

                page_key = f"page_{page_index + 1}"
                file_annotations[page_key] = {
                    "page_size": {"width": width, "height": height},
                    "annotations": page_ann_list,
                }

                buf = io.BytesIO()
                pil_page.save(buf, format="PNG")
                buf.seek(0)
                encoded = base64.b64encode(buf.read()).decode("utf-8")
                images_base64.append(encoded)

            doc.close()
        except Exception as e:
            return f"Failed to process PDF: {e}"

    # -----------------------------
    # Single image (PNG/JPG/...)
    # -----------------------------
    else:
        try:
            base_img = Image.open(filepath).convert("RGBA")
        except Exception:
            return "Only image and PDF formats are supported."

        width, height = base_img.size
        detections = detect_elements(base_img)
        draw = ImageDraw.Draw(base_img)
        colors = {
            "qr": (0, 128, 255, 255),
            "signature": (0, 200, 0, 255),
            "stamp": (255, 0, 0, 255),
        }
        page_ann_list = []

        for label, (x1, y1, x2, y2) in detections:
            if label == "qr" and not highlight_qr:
                continue
            if label == "signature" and not highlight_sig:
                continue
            if label == "stamp" and not highlight_stamp:
                continue

            color = colors.get(label, (255, 255, 0, 255))
            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=4)
            text = label.upper()
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except IOError:
                font = ImageFont.load_default()
            bbox_text = font.getbbox(text)
            tw, th = bbox_text[2] - bbox_text[0], bbox_text[3] - bbox_text[1]
            text_bg = (x1, y1 - th - 4, x1 + tw + 4, y1)
            bg_color = (*color[:3], 200)
            draw.rectangle(text_bg, fill=bg_color)
            draw.text(
                (x1 + 2, y1 - th - 2),
                text,
                fill=(255, 255, 255, 255),
                font=font,
            )

            # JSON annotation
            annotation_counter += 1
            w_box = x2 - x1
            h_box = y2 - y1
            ann_entry = {
                f"annotation_{annotation_counter}": {
                    "category": label,
                    "bbox": {
                        "x": float(x1),
                        "y": float(y1),
                        "width": float(w_box),
                        "height": float(h_box),
                    },
                    "area": float(w_box * h_box),
                }
            }
            page_ann_list.append(ann_entry)

        file_annotations["page_1"] = {
            "page_size": {"width": width, "height": height},
            "annotations": page_ann_list,
        }

        buf = io.BytesIO()
        base_img.save(buf, format="PNG")
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        images_base64.append(encoded)

    # -----------------------------
    # Save JSON annotations for this file
    # -----------------------------
    save_annotations_for_file(filename, file_annotations)

    # -----------------------------
    # Render result page
    # -----------------------------
    return render_template(
        "result.html",
        filename=filename,
        images=images_base64,
        files=files,
        prev_file=prev_file,
        next_file=next_file,
        highlight_qr=highlight_qr,
        highlight_sig=highlight_sig,
        highlight_stamp=highlight_stamp,
    )


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
