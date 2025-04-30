import cv2
import numpy as np
import os
from django.conf import settings

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


class FaceRecognizer:
    def __init__(self):
        self.reference_people = {}
        self.thresholds = {
            "admin": 5000,
            "p2": 4500
        }
        self.load_reference_faces()

    def load_reference_faces(self):
        """Load reference faces from the images folder"""
        base_dir = os.path.join(settings.BASE_DIR, 'images')

        for name in self.thresholds.keys():
            person_dir = os.path.join(base_dir, name)
            self.reference_people[name] = []

            if os.path.exists(person_dir):
                for filename in os.listdir(person_dir):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(person_dir, filename)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                        if img is not None:
                            faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
                            if len(faces) > 0:
                                x, y, w, h = faces[0]
                                face_img = cv2.resize(img[y:y + h, x:x + w], (200, 200))
                                self.reference_people[name].append(face_img)

    def process_image(self, image):
        """Process an image (file upload or numpy array)"""
        if isinstance(image, np.ndarray):
            return image

        file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
        return cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    def recognize_faces(self, image):
        """Recognize faces in an image"""
        gray = self.process_image(image)
        if gray is None:
            return {"error": "Could not process image"}

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        results = []

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            face_resized = cv2.resize(face_roi, (200, 200))

            best_match = "Unknown"
            best_score = float('inf')

            for name, ref_faces in self.reference_people.items():
                for ref_face in ref_faces:
                    score = np.mean((ref_face.astype("float") - face_resized.astype("float")) ** 2)
                    if score < best_score and score < self.thresholds.get(name, 5000):
                        best_score = score
                        best_match = name

            results.append({
                "position": {"x": x, "y": y, "w": w, "h": h},
                "identity": best_match,
                "score": float(best_score)
            })

        return {"faces": results} if results else {"error": "No faces detected"}