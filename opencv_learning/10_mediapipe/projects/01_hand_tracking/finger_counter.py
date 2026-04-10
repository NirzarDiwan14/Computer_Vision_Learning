import cv2
import mediapipe as mp
import time

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

# Finger tips
tipIds = [4, 8, 12, 16, 20]

# Hand connections
connections = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = landmarker.detect(mp_image)

    totalFingers = 0
    gesture = ""

    if result and result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:

            lmList = []

            # ---- Convert to pixel coords ----
            for id, lm in enumerate(hand_landmarks):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((cx, cy))

                cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

            # ---- Draw connections ----
            for connection in connections:
                x1, y1 = lmList[connection[0]]
                x2, y2 = lmList[connection[1]]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            fingers = []

            # ---- Thumb (works for both hands) ----
            if hand_landmarks[17].x < hand_landmarks[5].x:
                # Right hand
                if lmList[4][0] > lmList[3][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                # Left hand
                if lmList[4][0] < lmList[3][0]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # ---- Other fingers ----
            for i in range(1, 5):
                if lmList[tipIds[i]][1] < lmList[tipIds[i] - 2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            totalFingers = fingers.count(1)

            # ---- Gesture Recognition ----
            if fingers == [1, 0, 0, 0, 0]:
                gesture = "THUMBS UP"

            elif fingers == [0, 1, 1, 0, 0]:
                gesture = "COOL"

            elif fingers == [0, 1, 0, 0, 1]:
                gesture = "ROCK ON"

            else:
                gesture = ""

    # ---- UI ----
    cv2.rectangle(frame, (20, 200), (200, 400), (0, 255, 0), cv2.FILLED)

    cv2.putText(frame, str(totalFingers), (50, 350),
                cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 5)

    cv2.putText(frame, "FINGERS", (30, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # ---- Gesture Display ----
    if gesture != "":
        cv2.putText(frame, gesture, (200, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # ---- FPS ----
    cTime = time.time()
    fps = int(1 / (cTime - pTime)) if (cTime - pTime) != 0 else 0
    pTime = cTime

    cv2.putText(frame, f"FPS: {fps}", (400, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Gesture Recognition System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()