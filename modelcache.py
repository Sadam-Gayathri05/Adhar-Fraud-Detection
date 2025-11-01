from ultralytics import YOLO

_model = None

def get_yolo_model():
    global _model
    if _model is None:
        _model = YOLO("models/aadhar_yolo.pt")
    return _model
