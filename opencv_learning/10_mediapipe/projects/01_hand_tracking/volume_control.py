import cv2
import mediapipe as mp
import time
import math
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Load model
model_path = "/home/nirzar-diwan/Downloads/hand_landmarker.task"

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
RunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.IMAGE,
    num_hands=1,
)

landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

pTime = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = landmarker.detect(mp_image)

    if result and result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:

            # Landmarks
            thumb = hand_landmarks[4]
            index = hand_landmarks[8]

            x1, y1 = int(thumb.x * w), int(thumb.y * h)
            x2, y2 = int(index.x * w), int(index.y * h)

            # Draw points
            cv2.circle(frame, (x1, y1), 10, (0, 255, 255), -1)
            cv2.circle(frame, (x2, y2), 10, (255, 0, 255), -1)

            # Line
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Distance
            distance = math.hypot(x2 - x1, y2 - y1)

            # Map distance → volume (0–100)
            vol = int(distance / 200 * 100)
            vol = max(0, min(vol, 100))

            # Set volume (Linux)
            # os.system(f"pactl set-sink-volume @DEFAULT_SINK@ {vol}%")
            print(f"pactl set-sink-volume @DEFAULT_SINK@ {vol}%")

            # Display
            cv2.putText(frame, f"Volume: {vol}%", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Volume bar
            cv2.rectangle(frame, (50, 150), (85, 400), (255, 0, 0), 2)
            bar_height = int(400 - (vol / 100) * 250)
            cv2.rectangle(frame, (50, bar_height), (85, 400), (0, 255, 0), -1)

    # FPS
    cTime = time.time()
    fps = int(1 / (cTime - pTime)) if (cTime - pTime) != 0 else 0
    pTime = cTime

    cv2.putText(frame, f"FPS: {fps}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Volume Control", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()