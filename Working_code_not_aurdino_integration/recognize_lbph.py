import cv2
import face_recognition
import pickle
import time

# Load precomputed encodings
data = pickle.load(open("encodings.pkl", "rb"))
known_encodings = data["encodings"]
known_names = data["names"]

cap = cv2.VideoCapture(0)  # Webcam
THRESHOLD = 0.5  # Tune: 0.4-0.6 (lower = stricter)

# Optimization settings

FRAME_SKIP = 5  # Process every 5th frame (reduce load)
FRAME_WIDTH = 320  # Downscale frame width (faster processing)
FRAME_HEIGHT = 240  # Downscale frame height

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

print("Starting real-time recognition. Press 'q' to quit.")
frame_count = 0
last_label = "No Face Detected"
last_verdict = "DENIED"
current_face_locations = []  # Track current face locations for drawing

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Webcam not accessible.")
        break

    frame_count += 1
    if frame_count % FRAME_SKIP == 0:  # Process only every Nth frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")  # Use "hog" for speed
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        label = "No Face Detected"
        verdict = "DENIED"
        if face_encodings:
            distances = face_recognition.face_distance(known_encodings, face_encodings[0])
            min_distance = min(distances)
            index = distances.argmin()
            if min_distance < THRESHOLD:
                label = f"GRANTED: {known_names[index]} (dist: {min_distance:.2f})"
                verdict = "GRANTED"
            else:
                label = f"DENIED: Unknown (dist: {min_distance:.2f})"
                verdict = "DENIED"

        last_label = label
        last_verdict = verdict
        current_face_locations = face_locations  # Update for drawing

    # Draw red tracing box around detected faces
    for (top, right, bottom, left) in current_face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)  # Red box (BGR: 0,0,255)

    # Display the last computed result on all frames with smaller text
    cv2.putText(frame, last_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0) if last_verdict == "GRANTED" else (0, 0, 255), 2)
    cv2.imshow("ATM Face Verification", frame)

    print(last_verdict)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.05)  # Light sleep to prevent full CPU usage

cap.release()
cv2.destroyAllWindows()