import cv2
import os
import numpy as np
from deepface import DeepFace

# 1. Load reference images
reference_folder = "images"  # Folder with your reference photos
reference_encodings = {}

for filename in os.listdir(reference_folder):
    if filename.endswith((".jpg", ".png")):
        path = os.path.join(reference_folder, filename)
        img = cv2.imread(path)
        try:
            # Get face embedding using DeepFace
            embedding = DeepFace.represent(img, model_name="Facenet", enforce_detection=True)[0]["embedding"]
            reference_encodings[filename] = embedding
            print(f"Loaded: {filename}")
        except:
            print(f"No face detected in {filename}")

# 2. Capture from camera
cap = cv2.VideoCapture(0)
print("Press SPACE to capture, ESC to exit")

while True:
    ret, frame = cap.read()
    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break
    elif key == 32:  # SPACE
        try:
            # Get face embedding from camera
            camera_embedding = DeepFace.represent(frame, model_name="Facenet", enforce_detection=True)[0]["embedding"]

            # Compare with references
            for name, ref_embedding in reference_encodings.items():
                distance = np.linalg.norm(np.array(ref_embedding) - np.array(camera_embedding))
                print(f"Distance to {name}: {distance:.2f} (Lower = Better)")

                # Threshold: Typically < 0.6 is a match
                if distance < 0.6:
                    print(f"✅ MATCH: {name}")
                else:
                    print(f"❌ NO MATCH: {name}")

        except Exception as e:
            print(f"Error: {e}")

cap.release()
cv2.destroyAllWindows()