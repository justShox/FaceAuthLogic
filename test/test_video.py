import cv2
import os
import numpy as np

model_path = os.path.join("model", "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(model_path)

reference_folder = "images" # Dir for the the reference image
reference_faces = []
for filename in os.listdir(reference_folder):
    if filename.lower().endswith((".jpg", ".png")):
        img = cv2.imread(os.path.join(reference_folder, filename), cv2.IMREAD_GRAYSCALE)
        faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_img = cv2.resize(img[y:y + h, x:x + w], (200, 200))
            reference_faces.append(face_img)

if not reference_faces:
    print("Error: No reference faces loaded")
    exit()


def compare_faces(face1, face2):
    face1 = cv2.resize(face1, (200, 200))
    face2 = cv2.resize(face2, (200, 200))
    return np.mean((face1.astype("float") - face2.astype("float")) ** 2)


# Camera setup
cap = cv2.VideoCapture(0)
MATCH_THRESHOLD = 5000  # Adjust based on testing

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    # Process each face in frame
    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]

        # Compare with reference faces
        best_score = float('inf')
        for ref_face in reference_faces:
            score = compare_faces(face_roi, ref_face)
            if score < best_score:
                best_score = score

        # Draw rectangle and label
        color = (0, 255, 0) if best_score < MATCH_THRESHOLD else (0, 0, 255)
        label = "YOU" if best_score < MATCH_THRESHOLD else "Unknown"
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label} ({best_score:.0f})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Recognition - Press ESC", frame)
    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()