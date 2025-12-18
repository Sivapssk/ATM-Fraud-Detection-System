import cv2
import os

# Change 'user2' to the new user name
name = "varshitha"
out_dir = f"bank_faces/{name}"
os.makedirs(out_dir, exist_ok=True)

cap = cv2.VideoCapture(0)  # Webcam
i = 0
print("Press 's' to save each photo. Capture 100 images. Press 'q' to quit early.")

while i < 10:  # Increased for high accuracy
    ret, frame = cap.read()
    if not ret:
        print("Error: Webcam not accessible.")
        break
    cv2.imshow("Capture (press 's' to save)", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        cv2.imwrite(f"{out_dir}/{name}_{i}.jpg", frame)
        print(f"Saved {name}_{i}.jpg")
        i += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Done! Captured {i} images for {name}.")