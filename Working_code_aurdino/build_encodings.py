import face_recognition
import cv2
import os
import pickle
import numpy as np

DATA_DIR = "bank_faces"
AUGMENT_FACTOR = 2  # Augment each image (flip + brightness)

# Manually set the target user here (change this value as needed)
TARGET_USER = "varshitha"  # Example: Change to "user2" or "user3"

# Load existing encodings if file exists
if os.path.exists("encodings.pkl"):
    with open("encodings.pkl", "rb") as f:
        data = pickle.load(f)
    known_encodings = data["encodings"]
    known_names = data["names"]
    print(f"Loaded existing {len(known_encodings)} encodings for {len(set(known_names))} users.")
else:
    known_encodings = []
    known_names = []

# Check if target user already exists
if TARGET_USER in known_names:
    print(f"User '{TARGET_USER}' already exists. Skipping training.")
else:
    person_dir = os.path.join(DATA_DIR, TARGET_USER)
    if not os.path.isdir(person_dir):
        print(f"Folder '{person_dir}' not found. Create it and add images.")
    else:
        print(f"Processing new user: {TARGET_USER}")
        user_encodings = []
        for filename in os.listdir(person_dir):
            if filename.endswith(('.jpg', '.png')):
                filepath = os.path.join(person_dir, filename)
                img = cv2.imread(filepath)
                if img is None:
                    print(f"Warning: Could not load {filepath}")
                    continue
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Original image
                boxes = face_recognition.face_locations(rgb_img, model="cnn")  # High accuracy; use "hog" if slow
                if boxes:
                    encodings = face_recognition.face_encodings(rgb_img, boxes)
                    if encodings:
                        user_encodings.append(encodings[0])
                
                # Augmentation 1: Horizontal flip
                flipped = cv2.flip(rgb_img, 1)
                boxes_flip = face_recognition.face_locations(flipped, model="cnn")
                if boxes_flip:
                    encodings_flip = face_recognition.face_encodings(flipped, boxes_flip)
                    if encodings_flip:
                        user_encodings.append(encodings_flip[0])
                
                # Augmentation 2: Brightness adjustment (simulate lighting variation)
                bright_img = cv2.convertScaleAbs(rgb_img, alpha=1.2, beta=10)  # Increase brightness
                boxes_bright = face_recognition.face_locations(bright_img, model="hog")
                if boxes_bright:
                    encodings_bright = face_recognition.face_encodings(bright_img, boxes_bright)
                    if encodings_bright:
                        user_encodings.append(encodings_bright[0])
                
                print(f"Processed {filepath} with {AUGMENT_FACTOR} augmentations")

        # Add new user's encodings (multiple per image due to augmentation)
        for enc in user_encodings:
            known_encodings.append(enc)
            known_names.append(TARGET_USER)

# Save updated encodings
with open("encodings.pkl", "wb") as f:
    pickle.dump({"encodings": known_encodings, "names": known_names}, f)

print(f"Saved updated encodings.pkl with {len(known_encodings)} total encodings for {len(set(known_names))} users.")