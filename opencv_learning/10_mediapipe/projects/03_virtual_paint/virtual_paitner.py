import cv2
import numpy as np
import time
import mediapipe as mp
import math
from collections import deque

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ================== CONFIG ==================
WIDTH, HEIGHT = 1280, 720
HEADER_HEIGHT = 150

model_path = "/home/nirzar-diwan/Downloads/hand_landmarker.task"
header_path = "/home/nirzar-diwan/Downloads/paint_navbar.png"

# ================== MEDIAPIPE ==================
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
RunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6,
)

landmarker = HandLandmarker.create_from_options(options)


# ================== HEADER ==================
header_img = cv2.imread(header_path)
if header_img is None:
    print("⚠️  Navbar image not found — generating a default header.")
    header_img = np.zeros((HEADER_HEIGHT, WIDTH, 3), np.uint8)
    # Draw default colored buttons on fallback header
    colors_display = [
        (0, 250, (255, 0, 255)),
        (250, 500, (255, 0, 0)),
        (500, 750, (0, 255, 0)),
        (750, 1000, (0, 0, 255)),
        (1000, 1280, (50, 50, 50)),
    ]
    labels = ["Purple", "Blue", "Green", "Red", "Eraser"]
    for i, (x_start, x_end, color) in enumerate(colors_display):
        cv2.rectangle(header_img, (x_start, 0), (x_end, HEADER_HEIGHT), color, -1)
        cv2.putText(
            header_img, labels[i],
            (x_start + 10, HEADER_HEIGHT // 2 + 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )
else:
    header_img = cv2.resize(header_img, (WIDTH, HEADER_HEIGHT))


# ================== CAMERA ==================
cap = cv2.VideoCapture(0)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)


# ================== BUTTONS ==================
buttons = [
    (0,    250,  (255, 0, 255),  "Purple"),
    (250,  500,  (255, 0,   0),  "Blue"),
    (500,  750,  (0, 255,   0),  "Green"),
    (750,  1000, (0,   0, 255),  "Red"),
    (1000, 1280, (0,   0,   0),  "Eraser"),
]

# ================== STATE ==================
drawColor      = (255, 0, 255)
brushThickness = 12
eraserSize     = 60          # half-side of the square eraser

xp, yp         = 0, 0
canvas          = np.zeros((HEIGHT, WIDTH, 3), np.uint8)

# Undo stack — stores snapshots of canvas
undo_stack      = deque(maxlen=20)

pTime           = 0

# Landmark smoothing — running average of last N positions
SMOOTH_N        = 5
index_history   = deque(maxlen=SMOOTH_N)
thumb_history   = deque(maxlen=SMOOTH_N)


# ================== HELPERS ==================

def smooth_point(history, new_pt):
    history.append(new_pt)
    xs = [p[0] for p in history]
    ys = [p[1] for p in history]
    return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))


def fingers_up(hand, w, h):
    """
    Returns list [thumb, index, middle, ring, pinky] — 1 if finger is up, 0 if down.
    Uses Y-axis comparison: tip above PIP joint → finger is up.
    """
    tips  = [4, 8, 12, 16, 20]
    pips  = [3, 6, 10, 14, 18]   # one joint below the tip

    up = []
    # Thumb: compare x (left/right) depending on hand orientation
    # Simpler: just compare tip vs pip in X
    thumb_tip = hand[4]
    thumb_ip  = hand[3]
    # For a mirrored (flipped) feed, thumb is to the LEFT of IP when extended
    up.append(1 if thumb_tip.x < thumb_ip.x else 0)

    # Four fingers: tip Y < pip Y means finger is raised
    for tip_idx, pip_idx in zip(tips[1:], pips[1:]):
        up.append(1 if hand[tip_idx].y < hand[pip_idx].y else 0)

    return up  # [thumb, index, middle, ring, pinky]


def push_undo():
    undo_stack.append(canvas.copy())


def pop_undo():
    if undo_stack:
        return undo_stack.pop()
    return None


def draw_ui_overlay(frame, draw_color, brush_size, eraser_size, mode_text):
    """Draw overlay info on the bottom of the frame."""
    bar_y = HEIGHT - 40
    # Semi-transparent bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, bar_y), (WIDTH, HEIGHT), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Mode
    cv2.putText(frame, f"Mode: {mode_text}", (20, HEIGHT - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

    # Color swatch
    cv2.rectangle(frame, (300, bar_y + 5), (340, HEIGHT - 5), draw_color, -1)
    cv2.rectangle(frame, (300, bar_y + 5), (340, HEIGHT - 5), (255,255,255), 1)
    cv2.putText(frame, "Color", (350, HEIGHT - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Brush / eraser size
    size_label = f"Eraser: {eraser_size*2}px" if draw_color == (0,0,0) else f"Brush: {brush_size}px"
    cv2.putText(frame, size_label, (460, HEIGHT - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Undo hint
    cv2.putText(frame, "Z=Undo  Q=Quit", (WIDTH - 240, HEIGHT - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)


def highlight_selected_button(frame, draw_color):
    for x_start, x_end, color, _ in buttons:
        if color == draw_color:
            cv2.rectangle(frame, (x_start + 2, 2), (x_end - 2, HEADER_HEIGHT - 2),
                          (255, 255, 255), 4)
            break


# ================== MAIN LOOP ==================
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (WIDTH, HEIGHT))

    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    timestamp = int(time.time() * 1000)
    result    = landmarker.detect_for_video(mp_image, timestamp)

    mode_text = "Hover"

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        # Raw landmark positions
        raw_x1 = int(hand[8].x * WIDTH);  raw_y1 = int(hand[8].y * HEIGHT)   # index tip
        raw_xt = int(hand[4].x * WIDTH);  raw_yt = int(hand[4].y * HEIGHT)   # thumb tip
        raw_x2 = int(hand[12].x * WIDTH); raw_y2 = int(hand[12].y * HEIGHT)  # middle tip

        # Smoothed positions
        x1, y1 = smooth_point(index_history, (raw_x1, raw_y1))
        xt, yt = smooth_point(thumb_history,  (raw_xt, raw_yt))

        up = fingers_up(hand, WIDTH, HEIGHT)
        # up = [thumb, index, middle, ring, pinky]

        index_up  = up[1]
        middle_up = up[2]

        # ── SELECTION MODE: index + middle both up ──
        if index_up and middle_up:
            mode_text = "Select"
            xp, yp = 0, 0   # reset draw start

            # Visual: draw two circles and connecting line
            cv2.circle(frame, (x1, y1), 12, (0, 255, 255), cv2.FILLED)
            cv2.circle(frame, (raw_x2, raw_y2), 12, (0, 255, 255), cv2.FILLED)

            # Hover in header → select color
            if y1 < HEADER_HEIGHT:
                for x_start, x_end, color, _ in buttons:
                    if x_start < x1 < x_end:
                        drawColor = color

        # ── DRAW MODE: only index up ──
        elif index_up and not middle_up:
            mode_text = "Draw" if drawColor != (0, 0, 0) else "Erase"

            if drawColor == (0, 0, 0):
                # SQUARE ERASER — erase a rectangular region
                half = eraserSize
                x_min = max(0, x1 - half)
                x_max = min(WIDTH, x1 + half)
                y_min = max(0, y1 - half)
                y_max = min(HEIGHT, y1 + half)

                # Show eraser square on frame
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (200, 200, 200), 2)

                if xp != 0 and yp != 0:
                    push_undo()
                    # Erase along the path between prev and current point
                    steps = max(1, int(math.hypot(x1 - xp, y1 - yp) // (half // 2)))
                    for s in range(steps + 1):
                        t   = s / steps
                        ix  = int(xp + t * (x1 - xp))
                        iy  = int(yp + t * (y1 - yp))
                        ex1 = max(0, ix - half)
                        ex2 = min(WIDTH, ix + half)
                        ey1 = max(0, iy - half)
                        ey2 = min(HEIGHT, iy + half)
                        canvas[ey1:ey2, ex1:ex2] = 0

                xp, yp = x1, y1

            else:
                # BRUSH DRAWING
                cv2.circle(frame, (x1, y1), brushThickness, drawColor, cv2.FILLED)

                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                    push_undo()

                # Interpolate for smooth lines (fill gaps when moving fast)
                steps = max(1, int(math.hypot(x1 - xp, y1 - yp) // 4))
                for s in range(steps + 1):
                    t  = s / steps
                    ix = int(xp + t * (x1 - xp))
                    iy = int(yp + t * (y1 - yp))
                    cv2.circle(canvas, (ix, iy), brushThickness, drawColor, cv2.FILLED)

                cv2.line(canvas, (xp, yp), (x1, y1), drawColor, brushThickness * 2)
                xp, yp = x1, y1

        else:
            # No relevant gesture → lift pen
            xp, yp = 0, 0

    else:
        # No hand detected
        xp, yp = 0, 0
        index_history.clear()
        thumb_history.clear()


    # ================== MERGE CANVAS ==================
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    mask_inv_3ch = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)
    frame_bg     = cv2.bitwise_and(frame, mask_inv_3ch)
    frame        = cv2.bitwise_or(frame_bg, canvas)


    # ================== HEADER ==================
    frame[0:HEADER_HEIGHT, 0:WIDTH] = header_img

    # Highlight active color button
    highlight_selected_button(frame, drawColor)

    # Draw button border outlines
    for x_start, x_end, color, _ in buttons:
        cv2.rectangle(frame, (x_start, 0), (x_end, HEADER_HEIGHT), color, 2)


    # ================== FPS ==================
    cTime = time.time()
    fps   = int(1 / (cTime - pTime + 1e-9))
    pTime = cTime
    cv2.putText(frame, f"FPS: {fps}", (WIDTH - 120, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


    # ================== UI OVERLAY ==================
    draw_ui_overlay(frame, drawColor, brushThickness, eraserSize, mode_text)


    cv2.imshow("Virtual Painter", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("z") or key == ord("Z"):
        restored = pop_undo()
        if restored is not None:
            canvas = restored
    elif key == ord("+") or key == ord("="):
        brushThickness = min(50, brushThickness + 2)
        eraserSize     = min(120, eraserSize + 10)
    elif key == ord("-"):
        brushThickness = max(2, brushThickness - 2)
        eraserSize     = max(20, eraserSize - 10)
    elif key == ord("c"):
        push_undo()
        canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)


cap.release()
cv2.destroyAllWindows()