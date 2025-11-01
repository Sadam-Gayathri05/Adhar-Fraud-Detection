# fraudchecker.py
import cv2
import numpy as np

def check_fraud(image):
    """
    Return heuristic fraud_score in [0,1].
    Replace with a trained model for production.
    """
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # edge density (text-like structure)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / (h * w + 1e-9)

    # contrast measure
    contrast = gray.std() / 255.0

    score = 0.0
    if edge_density < 0.02:
        score += 0.6
    if contrast < 0.08:
        score += 0.4

    score = max(0.0, min(round(score, 3), 1.0))
    return score
