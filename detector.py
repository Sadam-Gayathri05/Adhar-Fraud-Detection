import os
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
yolo_model = YOLO(os.path.join("models", "aadhaar_yolo.pt"))

def detect_aadhaar_boxes(image_path):
    """
    Detect Aadhaar card bounding boxes using YOLOv8.
    Returns (bounding_boxes, cropped_image)
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, None

    results = yolo_model.predict(img)

    if results[0].boxes:
        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype(int)

        # Take the first bounding box and expand margin slightly
        x1, y1, x2, y2 = boxes_xyxy[0]
        h, w = img.shape[:2]
        margin_x = int((x2 - x1) * 0.05)
        margin_y = int((y2 - y1) * 0.05)
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(w - 1, x2 + margin_x)
        y2 = min(h - 1, y2 + margin_y)

        cropped = img[y1:y2, x1:x2].copy()
    else:
        cropped = img.copy()  # fallback to full image
        boxes_xyxy = np.array([])

    return boxes_xyxy, cropped
