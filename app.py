import os
import cv2
import re
import pytesseract
import pandas as pd
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime
import numpy as np

# ---------------- CONFIG ----------------
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
RESULTS_FOLDER = "static/results"
CSV_PATH = os.path.join(RESULTS_FOLDER, "extracted_details.csv")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# adjust to your tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Tesseract config (PSM 6 = assume a single uniform block of text)
TESSERACT_CONFIG = r"--psm 6"

# ---------------- Verhoeff for Aadhaar validation ----------------
# Verhoeff algorithm tables
_d = [
    [0,1,2,3,4,5,6,7,8,9],
    [1,2,3,4,0,6,7,8,9,5],
    [2,3,4,0,1,7,8,9,5,6],
    [3,4,0,1,2,8,9,5,6,7],
    [4,0,1,2,3,9,5,6,7,8],
    [5,9,8,7,6,0,4,3,2,1],
    [6,5,9,8,7,1,0,4,3,2],
    [7,6,5,9,8,2,1,0,4,3],
    [8,7,6,5,9,3,2,1,0,4],
    [9,8,7,6,5,4,3,2,1,0]
]

_p = [
    [0,1,2,3,4,5,6,7,8,9],
    [1,5,7,6,2,8,3,0,9,4],
    [5,8,0,3,7,9,6,1,4,2],
    [8,9,1,6,0,4,3,5,2,7],
    [9,4,5,3,1,2,6,8,7,0],
    [4,2,8,6,5,7,3,9,0,1],
    [2,7,9,3,8,0,6,4,1,5],
    [7,0,4,6,9,1,3,2,5,8]
]

_inv = [0,4,3,2,1,5,6,7,8,9]

def verhoeff_validate(number_str: str) -> bool:
    """Return True if number_str passes Verhoeff checksum."""
    try:
        c = 0
        # process digits from right to left
        for i, ch in enumerate(reversed(number_str)):
            d = ord(ch) - 48
            c = _d[c][_p[(i % 8)][d]]
        return c == 0
    except Exception:
        return False

# ---------------- OCR / PREPROCESS ----------------
def preprocess_image(image):
    """Convert to gray, denoise, morphological cleaning, and adaptive threshold.
    Returns processed grayscale image (for OCR) and a slightly enhanced color copy for drawing boxes.
    """
    img = image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Contrast limited adaptive histogram equalization (CLAHE) sometimes helps
    try:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
    except Exception:
        pass

    # bilateral filter preserves edges
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # morphological opening to remove small spots
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=1)

    # adaptive threshold (if image is too dark/bright, this can help)
    try:
        gray_t = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2
        )
        gray = gray_t
    except Exception:
        pass

    return gray, img

def safe_conf_to_float(conf):
    try:
        # tesseract returns confidences as strings like "-1" or "85"
        return float(conf)
    except Exception:
        return -1.0

# ---------------- EXTRACTION ----------------
def extract_aadhaar_details(image):
    """Extract text fields, draw bounding boxes on a copy and return structured data."""
    gray, draw_img = preprocess_image(image)

    # get OCR data with bounding boxes
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, config=TESSERACT_CONFIG)
    n_boxes = len(data.get("text", []))

    # collect words and confidences; also draw boxes for words with decent confidence
    words = []
    for i in range(n_boxes):
        txt = data["text"][i].strip()
        conf = safe_conf_to_float(data["conf"][i])
        if txt != "":
            words.append(txt)
        # draw boxes only for good confidences to avoid clutter
        if conf > 60:
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            cv2.rectangle(draw_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Build flattened text for regex matching. Keep punctuation that might separate digits.
    text = " ".join(words).upper()

    # Aadhaar is typically 12 digits often grouped as 4 4 4 or with dashes/spaces. Accept digits only.
    aadhaar_search = re.search(r'(\d{4}\s*\d{4}\s*\d{4}|\d{12})', text)
    aadhaar_raw = aadhaar_search.group() if aadhaar_search else None
    aadhaar_digits = None
    if aadhaar_raw:
        # remove all non-digit chars
        aadhaar_digits = re.sub(r'\D', '', aadhaar_raw)

    # DOB: accept dd/mm/yyyy or dd-mm-yyyy or dd.mm.yyyy or yyyy-mm-dd
    dob_search = re.search(r'\b(\d{2}[\/\-\.\s]\d{2}[\/\-\.\s]\d{4}|\d{4}[\/\-\.\s]\d{2}[\/\-\.\s]\d{2})\b', text)
    dob = dob_search.group() if dob_search else None

    # Gender detection (look for full words or single letters near words like SEX/GENDER)
    gender = None
    if re.search(r'\bMALE\b', text):
        gender = "Male"
    elif re.search(r'\bFEMALE\b', text):
        gender = "Female"
    else:
        # single letter M or F near keywords
        m = re.search(r'(?:SEX|GENDER)[\:\s]*([MF])\b', text)
        if m:
            gender = "Male" if m.group(1) == "M" else "Female"
        else:
            # fallback: solitary 'M' or 'F' tokens (risky)
            if re.search(r'\bM\b', text) and not re.search(r'\bMR\b', text):
                gender = "Male"
            elif re.search(r'\bF\b', text) and not re.search(r'\bF/O\b', text):
                gender = "Female"

    # Name extraction: Aadhaar often has name lines in uppercase; try to find probable name
    # Heuristic: line with 2-4 consecutive capitalized words (could include initials)
    name = None
    # try to find patterns like "FIRST LAST" or "FIRST MIDDLE LAST"
    name_match = re.findall(r'\b([A-Z][A-Z\.]{1,}\s(?:[A-Z][A-Z\.]{1,}\s?){0,3})\b', text)
    # If above returns nothing, fallback to sequences of 2-3 uppercase words
    if not name_match:
        name_match = re.findall(r'\b([A-Z]{2,}\s[A-Z]{2,}(?:\s[A-Z]{2,})?)\b', text)

    if name_match:
        # choose the longest candidate (most likely full name)
        name = max(name_match, key=lambda s: len(s)).strip()

    return {
        "aadhaar_number_raw": aadhaar_raw,
        "aadhaar_number_digits": aadhaar_digits,
        "aadhaar_valid_verhoeff": verhoeff_validate(aadhaar_digits) if aadhaar_digits and len(aadhaar_digits)==12 else False,
        "dob": dob,
        "gender": gender,
        "name": name,
        "raw_text": text,
        "boxed_image": draw_img,
        "ocr_word_count": len(words)
    }

def is_aadhaar_card(text):
    # if contains the word 'AADHAAR' or a plausible 12-digit sequence
    if text is None:
        return False
    if re.search(r'\bAADHAAR\b', text) or re.search(r'\b\d{4}\s*\d{4}\s*\d{4}\b', text):
        return True
    return False

def check_fraud(details):
    """Return fraud score 0..1 (1 => high suspicion). Weighted heuristics:
       - If Aadhaar digits found but fail Verhoeff -> large penalty
       - Missing aadhaar number -> moderate penalty
       - Missing/short name -> small penalty
       - Missing gender or dob -> small penalty
    """
    score = 0.0

    # If digits found but not 12 digits -> suspicious
    if details.get("aadhaar_number_digits"):
        if len(details["aadhaar_number_digits"]) != 12:
            score += 0.45
        else:
            # 12 digits: if verhoeff fails -> big penalty
            if not details.get("aadhaar_valid_verhoeff", False):
                score += 0.85
            else:
                # valid number reduces suspicion substantially
                score += 0.0
    else:
        # no digits found at all
        score += 0.6

    # name check
    if not details.get("name"):
        score += 0.2
    else:
        if len(details["name"].split()) < 2:
            # single token name is less reliable
            score += 0.1

    # DOB & gender checks
    if not details.get("dob"):
        score += 0.1
    if not details.get("gender"):
        score += 0.05

    # clamp to 1.0
    return min(score, 1.0)

# ---------------- ROUTES ----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty file"}), 400

    filename = secure_filename(file.filename)
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(image_path)

    image = cv2.imread(image_path)
    if image is None:
        return jsonify({"error": "Cannot read image"}), 400

    details = extract_aadhaar_details(image)
    valid_aadhaar_text = is_aadhaar_card(details["raw_text"])
    fraud_score = check_fraud(details)

    # Use both heuristic label and verhoeff: if verhoeff OK and low fraud_score -> Valid Aadhaar
    label = "Fake Aadhaar"
    if details.get("aadhaar_number_digits") and details.get("aadhaar_valid_verhoeff") and fraud_score < 0.4:
        label = "Valid Aadhaar"
    else:
        # if textual cues strongly indicate Aadhaar and verhoeff passes, prefer valid
        if valid_aadhaar_text and details.get("aadhaar_number_digits") and details.get("aadhaar_valid_verhoeff"):
            label = "Valid Aadhaar"

    # Save boxed image
    boxed_filename = f"boxed_{filename}"
    boxed_path = os.path.join(RESULTS_FOLDER, boxed_filename)
    cv2.imwrite(boxed_path, details["boxed_image"])

    # Save results
    record = {
        "filename": filename,
        "aadhaar_number_raw": details.get("aadhaar_number_raw") or "Not Found",
        "aadhaar_digits": details.get("aadhaar_number_digits") or "Not Found",
        "aadhaar_verhoeff_valid": bool(details.get("aadhaar_valid_verhoeff")),
        "name": details.get("name") or "Not Found",
        "dob": details.get("dob") or "Not Found",
        "gender": details.get("gender") or "Not Found",
        "fraud_score": fraud_score,
        "label": label,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    df = pd.DataFrame([record])
    if os.path.exists(CSV_PATH):
        old = pd.read_csv(CSV_PATH)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(CSV_PATH, index=False)

    return render_template(
        "result.html",
        details=record,
        boxed_image=f"/{boxed_path.replace(os.sep, '/')}",
        original_image=f"/{image_path.replace(os.sep, '/')}",
    )

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)
