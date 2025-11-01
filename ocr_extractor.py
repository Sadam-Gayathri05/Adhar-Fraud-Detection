# ocr_extractor.py
import cv2
import re
import pytesseract

# Path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_aadhaar_details(image):
    """
    Extract Aadhaar card details using OCR with preprocessing.
    Returns a dictionary with aadhaar_no, name, dob, gender, and raw_text.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Remove noise and preserve edges
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # Adaptive thresholding for better text segmentation
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)

    # Optional: Morphological closing to enhance characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # OCR with Tesseract
    text = pytesseract.image_to_string(gray, config='--psm 6')
    text = text.replace("\n", " ").strip()

    # Extract Aadhaar number (12-digit format)
    aadhaar_match = re.search(r'\b\d{4}\s\d{4}\s\d{4}\b', text)
    aadhaar_no = aadhaar_match.group() if aadhaar_match else ""

    # Extract DOB: DD/MM/YYYY or DD-MM-YYYY
    dob_match = re.search(r'\b\d{2}[/-]\d{2}[/-]\d{4}\b', text)
    dob = dob_match.group() if dob_match else ""

    # Extract Gender
    gender = ""
    if re.search(r'\bMale\b', text, re.IGNORECASE):
        gender = "Male"
    elif re.search(r'\bFemale\b', text, re.IGNORECASE):
        gender = "Female"

    # Extract Name: first line that is not gender, dob, or Aadhaar number
    lines = [line.strip() for line in text.split() if line.strip()]
    name = ""
    for line in lines:
        if line not in [dob, gender] and not re.match(r'\d{4}\s\d{4}\s\d{4}', line):
            name = line
            break

    details = {
        "aadhaar_no": aadhaar_no,
        "name": name,
        "dob": dob,
        "gender": gender,
        "raw_text": text
    }
    return details


def is_aadhaar_card(text):
    """
    Basic Aadhaar validity check: presence of 12-digit number
    """
    return bool(re.search(r'\b\d{4}\s\d{4}\s\d{4}\b', text))
