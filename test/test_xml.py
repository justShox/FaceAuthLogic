# import cv2
# import os
# import numpy as np
#
# # 1. Load Haar Cascade from your model folder
# model_path = os.path.join("model", "haarcascade_frontalface_default.xml")
# if not os.path.exists(model_path):
#     print(f"Error: XML file not found at {model_path}")
#     exit()
#
# face_cascade = cv2.CascadeClassifier(model_path)
# if face_cascade.empty():
#     print("Error: Failed to load Haar Cascade classifier")
#     exit()
#
# # 2. Load reference images from 'images' folder
# reference_folder = "images"
# if not os.path.exists(reference_folder):
#     print(f"Error: '{reference_folder}' folder not found")
#     exit()
#
# reference_faces = {}
# print("Loading reference images...")
#
# for filename in os.listdir(reference_folder):
#     if filename.lower().endswith((".jpg", ".jpeg", ".png")):
#         path = os.path.join(reference_folder, filename)
#         img = cv2.imread(path)
#         if img is None:
#             print(f"Warning: Could not read {filename}")
#             continue
#
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
#
#         if len(faces) > 0:
#             (x, y, w, h) = faces[0]
#             face_roi = gray[y:y + h, x:x + w]
#             face_resized = cv2.resize(face_roi, (200, 200))
#             reference_faces[filename] = face_resized
#             print(f"✔ Loaded {filename} (Face detected)")
#         else:
#             print(f"✖ Skipped {filename} (No face found)")
#
# if not reference_faces:
#     print("Error: No valid reference images with faces found!")
#     exit()
#
# # 3. Camera capture and comparison
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Error: Could not open camera")
#     exit()
#
# print("\nPress SPACE to capture, ESC to exit")
# print("Make sure your face is well-lit and centered")
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Camera frame not available")
#         break
#
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     cv2.imshow("Camera - Press SPACE", frame)
#
#     key = cv2.waitKey(1)
#     if key == 27:  # ESC
#         break
#     elif key == 32:  # SPACE
#         faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
#
#         if len(faces) == 0:
#             print("Error: No face detected in camera!")
#             continue
#
#         (x, y, w, h) = faces[0]
#         camera_face = gray_frame[y:y + h, x:x + w]
#         camera_face = cv2.resize(camera_face, (200, 200))
#
#         best_match = None
#         best_score = float('inf')
#
#         for name, ref_face in reference_faces.items():
#             mse = np.mean((ref_face.astype("float") - camera_face.astype("float")) ** 2)
#             print(f"Difference with {name}: {mse:.1f} (Lower = Better)")
#
#             if mse < best_score:
#                 best_score = mse
#                 best_match = name
#
#         MATCH_THRESHOLD = 5000  # Adjust this value as needed
#         if best_score < MATCH_THRESHOLD:
#             print(f"✅ MATCH: {best_match} (Score: {best_score:.1f})")
#         else:
#             print(f"❌ NO MATCH (Best score: {best_score:.1f})")
#
# cap.release()
# cv2.destroyAllWindows()

import cv2
import os
import numpy as np

# Load Haar Cascade
model_path = os.path.join("model", "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(model_path)

reference_people = {
    "admin": [],
    "p2": [],
    # "Bob": [],
}

for name in reference_people.keys():
    person_folder = os.path.join("images", name)
    if not os.path.exists(person_folder):
        print(f"Warning: No folder for {name}")
        continue

    for filename in os.listdir(person_folder):
        if filename.lower().endswith((".jpg", ".png")):
            img = cv2.imread(os.path.join(person_folder, filename), cv2.IMREAD_GRAYSCALE)
            faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face_img = cv2.resize(img[y:y + h, x:x + w], (200, 200))
                reference_people[name].append(face_img)

# Thresholds per person (customizable)
MATCH_THRESHOLDS = {
    "admin": 5000,
    "p2": 4500,
    # "Bob": 5500
}

if not reference_people:
    print("Error: No reference faces found in 'images' folder")
    exit()

cap = cv2.VideoCapture(0)
print("Taking photo in 3 seconds... (Look at the camera)")
for i in range(3, 0, -1):
    print(f"{i}...", end=' ', flush=True)
    cv2.waitKey(1000)  # Wait 1 second per count

ret, frame = cap.read()
cap.release()

if not ret:
    print("\nError: Failed to capture photo")
    exit()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

if len(faces) == 0:
    print("\nResult: No faces detected in photo")
    exit()

MATCH_THRESHOLD = 5000 # Fix depending on ur instructions (the lower - more likely u!)
print("\n--- Results ---")

for i, (x, y, w, h) in enumerate(faces):
    face_roi = gray[y:y + h, x:x + w]
    face_resized = cv2.resize(face_roi, (200, 200))

    best_score = float('inf')
    for ref_face in reference_people:
        score = np.mean((ref_face.astype("float") - face_resized.astype("float")) ** 2)
        if score < best_score:
            best_score = score

    if best_score < MATCH_THRESHOLD:
        print(f"Face {i + 1}: ✅ YOU (Score: {best_score:.0f})")
    else:
        print(f"Face {i + 1}: ❌ Unknown (Score: {best_score:.0f})")

# Optional: Save the captured photo
cv2.imwrite("captured_photo.jpg", frame)
print("Photo saved as 'captured_photo.jpg'")