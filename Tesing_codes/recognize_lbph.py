import cv2

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("lbph_model.yml")

# Load label map
label_map = {}
with open("label_map.txt", "r") as f:
    for line in f:
        id, name = line.strip().split(":")
        label_map[int(id)] = name

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
CONFIDENCE_THRESHOLD = 50  # Lower = stricter (tune 40-60)

print("Starting LBPH recognition. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    label = "No Face Detected"
    verdict = "DENIED"
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face_crop = gray[y:y+h, x:x+w]
        id, confidence = recognizer.predict(face_crop)
        if confidence < CONFIDENCE_THRESHOLD:
            name = label_map.get(id, "Unknown")
            label = f"GRANTED: {name} (conf: {confidence:.0f})"
            verdict = "GRANTED"
        else:
            label = f"DENIED: Unknown (conf: {confidence:.0f})"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0) if verdict == "GRANTED" else (0, 0, 255), 2)

    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 255, 0) if verdict == "GRANTED" else (0, 0, 255), 2)
    cv2.imshow("ATM Face Verification (LBPH)", frame)

    print(verdict)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
