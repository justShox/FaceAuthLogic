# import cv2
# import numpy as np
# import face_recognition
# from django.core.files.uploadedfile import InMemoryUploadedFile
#
#
# def process_image(image):
#     # Convert to OpenCV format
#     if isinstance(image, InMemoryUploadedFile):
#         file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
#         img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
#     else:
#         img = cv2.imread(image)
#
#     # Convert to RGB (face_recognition uses RGB)
#     rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     return rgb_img
#
#
# def get_face_encodings(image):
#     rgb_img = process_image(image)
#
#     # Find all face locations
#     face_locations = face_recognition.face_locations(rgb_img)
#
#     if not face_locations:
#         return None
#
#     # Get encodings for all faces found
#     face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
#     return face_encodings
#
#
# def compare_faces(known_encoding, unknown_encoding, tolerance=0.6):
#     if not known_encoding or not unknown_encoding:
#         return False
#
#     # Compare faces
#     results = face_recognition.compare_faces(
#         [known_encoding],
#         unknown_encoding,
#         tolerance=tolerance
#     )
#     return results[0]


# recognition/utils.py
from deepface import DeepFace
import cv2
import numpy as np
from django.core.files.uploadedfile import InMemoryUploadedFile


def process_image(image):
    """Handle all input types: file uploads, paths, and numpy arrays"""
    if isinstance(image, np.ndarray):  # Camera frame (numpy array)
        return image

    if hasattr(image, 'read'):  # File upload object
        file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
        return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if isinstance(image, str):  # File path
        return cv2.imread(image)

    raise ValueError("Unsupported image input type")


def get_face_encodings(image):
    """Process image and extract face encodings"""
    img = process_image(image)
    if img is None:
        return None

    # Convert to grayscale for Haar Cascade
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) == 0:
        return None

    # Create encodings for each face
    face_encodings = []
    for (x, y, w, h) in faces:
        face_region = gray[y:y + h, x:x + w]
        face_resized = cv2.resize(face_region, (100, 100))
        face_encodings.append(face_resized.flatten())

    return face_encodings


def compare_faces(known_encoding, unknown_encoding, tolerance=0.6):
    """Compare embeddings using cosine similarity"""
    if not known_encoding or not unknown_encoding:
        return False

    # Convert bytes back to numpy array if needed
    if isinstance(known_encoding, bytes):
        known_encoding = np.frombuffer(known_encoding, dtype=np.float32)
    if isinstance(unknown_encoding, bytes):
        unknown_encoding = np.frombuffer(unknown_encoding, dtype=np.float32)

    # Cosine similarity (higher = more similar)
    similarity = np.dot(known_encoding, unknown_encoding) / (
            np.linalg.norm(known_encoding) * np.linalg.norm(unknown_encoding))

    return similarity > (1 - tolerance)  # Threshold adjustment