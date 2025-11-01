import cv2
import face_recognition

def verify_face(image, known_face_path="known_face.jpg"):
    """
    Compare face on Aadhaar with known face image.
    Returns match score (0-1)
    """
    try:
        known_image = face_recognition.load_image_file(known_face_path)
        known_encoding = face_recognition.face_encodings(known_image)[0]

        # Detect face in Aadhaar card
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(image_rgb)

        if not encodings:
            return 0  # No face found

        # Compare
        matches = face_recognition.compare_faces([known_encoding], encodings[0])
        return 1 if matches[0] else 0
    except Exception as e:
        print("Face verification error:", e)
        return 0
