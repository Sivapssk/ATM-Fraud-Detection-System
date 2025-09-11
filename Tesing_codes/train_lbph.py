import cv2
import os
import numpy as np

DATA_DIR = "bank_faces"
recognizer = cv2.face.LBPHFaceRecognizer_create()  # Should work with contrib
faces = []
labels = []
label_map = {}  # name to ID
id_counter = 0

for person in os.listdir(DATA_DIR):
    person_dir = os.path.join(DATA_DIR, person)
    if os.path.isdir(person_dir):
        label_map[id_counter] = person
        for filename in os.listdir(person_dir):
            if filename.endswith(('.jpg', '.png')):
                filepath = os.path.join(person_dir, filename)
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # Grayscale for LBPH
                if img is None:
                    print(f"Warning: Could not load {filepath}")
                    continue
                # Detect face (simple Haar cascade)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                detected_faces = face_cascade.detectMultiScale(img, 1.1, 4)
                if len(detected_faces) > 0:
                    (x, y, w, h) = detected_faces[0]
                    face_crop = img[y:y+h, x:x+w]
                    faces.append(face_crop)
                    labels.append(id_counter)
        id_counter += 1

if not faces:
    print("No faces found. Check dataset.")
else:
    recognizer.train(faces, np.array(labels))
    recognizer.save("lbph_model.yml")
    with open("label_map.txt", "w") as f:
        for id, name in label_map.items():
            f.write(f"{id}:{name}\n")
    print("Trained and saved lbph_model.yml")
