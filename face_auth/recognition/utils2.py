import cv2
import numpy as np
from django.core.files.uploadedfile import InMemoryUploadedFile

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def process_image(image):
    if isinstance(image, InMemoryUploadedFile):
        file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(image)
    return img


def get_face_encodings(image):
    img = process_image(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    # For simplicity, we'll just return the face coordinates
    # In a real app, you'd want to generate proper encodings
    face_encodings = []
    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face_resized = cv2.resize(face, (100, 100))  # Standard size
        face_encodings.append(face_resized.flatten())  # Simple encoding

    return face_encodings


def compare_faces(known_encoding, unknown_encoding, tolerance=0.8):
    if not known_encoding or not unknown_encoding:
        return False

    # Simple comparison using MSE (Mean Squared Error)
    mse = np.mean((known_encoding - unknown_encoding) ** 2)
    return mse < (1 - tolerance) * 10000  # Adjust threshold as needed