import cv2
import mediapipe as mp
import time
import math

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

# Cursor smoothing
prev_x, prev_y = 0, 0
smoothening = 5

# Click control
clicking = False

# FPS
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
            index_tip = hand_landmarks[8]
            thumb_tip = hand_landmarks[4]

            ix, iy = int(index_tip.x * w), int(index_tip.y * h)
            tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)

            # Smooth cursor
            curr_x = prev_x + (ix - prev_x) // smoothening
            curr_y = prev_y + (iy - prev_y) // smoothening

            prev_x, prev_y = curr_x, curr_y

            # Draw cursor & thumb
            cv2.circle(frame, (curr_x, curr_y), 15, (255, 0, 255), -1)
            cv2.circle(frame, (tx, ty), 10, (0, 255, 255), -1)

            # Distance (click detection)
            distance = math.hypot(
                thumb_tip.x - index_tip.x,
                thumb_tip.y - index_tip.y
            )

            # Draw line between fingers
            cv2.line(frame, (ix, iy), (tx, ty), (0, 255, 0), 2)

            # Click logic
            if distance < 0.05:
                if not clicking:
                    clicking = True
                    cv2.putText(frame, "CLICK", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                clicking = False

            # ---- BUTTON (MID-LEFT SMALL) ----
            btn_x1 = 50
            btn_y1 = h // 2 - 40
            btn_x2 = 150
            btn_y2 = h // 2 + 40

            if btn_x1 < curr_x < btn_x2 and btn_y1 < curr_y < btn_y2:
                cv2.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), (0, 255, 0), -1)

                if clicking:
                    cv2.putText(frame, "PRESSED", (btn_x1, btn_y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                cv2.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), (255, 0, 0), 2)

    # ---- FPS ----
    cTime = time.time()
    fps = int(1 / (cTime - pTime)) if (cTime - pTime) != 0 else 0
    pTime = cTime

    cv2.putText(frame, f"FPS: {fps}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Virtual Cursor System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()