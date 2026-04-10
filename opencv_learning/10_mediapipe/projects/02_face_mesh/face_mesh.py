import cv2
import mediapipe as mp
import time

# NEW API imports
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Load model (download from mediapipe official repo)

model_path = "/home/nirzar-diwan/Downloads/face_landmarker.task"

# Create Face Landmarker
BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

# Callback function (required for LIVE_STREAM mode)
def print_result(result, output_image, timestamp_ms):
    global face_result
    face_result = result

face_result = None


options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_faces=2,
    result_callback=print_result, 
)


landmarker = FaceLandmarker.create_from_options(options)

# Webcam
cap = cv2.VideoCapture(0)
pTime = 0

while True:
    success, frame = cap.read()
    if not success:
        print("Could not read frame")
        break

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to mediapipe image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Send frame to model
    timestamp = int(time.time() * 1000)
    landmarker.detect_async(mp_image, timestamp)

    # Draw landmarks if available
    if face_result and face_result.face_landmarks:
        for face_landmarks in face_result.face_landmarks:
            
            h, w, _ = frame.shape

            # Example points
            nose = face_landmarks[1]
            left_eye = face_landmarks[33]
            right_eye = face_landmarks[263]
            chin = face_landmarks[152]
            nose = face_landmarks[1]
            
            

            # Head direction detection
            left_eye = face_landmarks[33]
            right_eye = face_landmarks[263]

            # convert to pixels
            nx, ny = int(nose.x*w), int(nose.y*h)
            lx, ly = int(left_eye.x*w), int(left_eye.y*h)
            rx, ry = int(right_eye.x*w), int(right_eye.y*h)

            # Distance from camera detection
            eye_dist = abs(lx - rx)

            if eye_dist > 120:
                text = "VERY CLOSE"
            elif eye_dist > 80:
                text = "CLOSE"
            elif eye_dist > 50:
                text = "NORMAL"
            else:
                text = "FAR"

            cv2.putText(frame, text, (30, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

            # distance
            dist_left = abs(nx - lx)
            dist_right = abs(nx - rx)

            if dist_left > dist_right + 10:
                text = "Looking Right"
            elif dist_right > dist_left + 10:
                text = "Looking Left"
            else:
                text = "Looking Center"

            cv2.putText(frame, text, (30,100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
            # Eye open/close detection
            top = face_landmarks[159]
            bottom = face_landmarks[145]

            ty = int(top.y * h)
            by = int(bottom.y * h)

            eye_open = abs(ty - by)

            if eye_open < 5:
                cv2.putText(frame, "BLINK", (30,200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            else:
                cv2.putText(frame, "NOT BLINK", (30,200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

            # Mouth open/close detection
            upper_lip = face_landmarks[13]
            lower_lip = face_landmarks[14]

            uy = int(upper_lip.y * h)
            ly = int(lower_lip.y * h)

            mouth_open = abs(ly - uy)

            if mouth_open > 15:
                cv2.putText(frame, "MOUTH OPEN", (30,150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            else:
                cv2.putText(frame, "MOUTH CLOSE", (30,150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)


            points = [nose, left_eye, right_eye, chin]

            for lm in points:
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # FPS
    cTime = time.time()
    fps = int(1 / (cTime - pTime)) if (cTime - pTime) != 0 else 0
    pTime = cTime

    cv2.putText(frame, f"FPS {fps}", (20, 70),
                cv2.FONT_HERSHEY_PLAIN, 1.5, (240, 0, 0), 3)

    cv2.imshow("Webcam Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()