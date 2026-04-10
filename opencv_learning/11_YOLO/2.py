from ultralytics import YOLO
import cv2

# Load model
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

# Line position (door area)
line_y = 300

# Track previous positions
person_positions = {}

person_id = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.resize(frame, (800, 600))

    # Run detection
    results = model(frame)

    boxes = results[0].boxes

    current_positions = {}

    if boxes is not None:
        for i, box in enumerate(boxes):
            cls = int(box.cls[0])

            # Only detect PERSON (class 0 in COCO)
            if cls == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                current_positions[i] = cy

                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.circle(frame, (cx, cy), 5, (0,0,255), -1)

                # Check movement direction
                if i in person_positions:
                    prev_y = person_positions[i]

                    # Crossing DOWN → IN
                    if prev_y < line_y and cy > line_y:
                        print("✅ Person went IN")

                    # Crossing UP → OUT
                    elif prev_y > line_y and cy < line_y:
                        print("⬅️ Person went OUT")

    # Update positions
    person_positions = current_positions

    # Draw door line
    cv2.line(frame, (0, line_y), (800, line_y), (255,0,0), 3)
    cv2.putText(frame, "DOOR LINE", (10, line_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

    cv2.imshow("Entry Exit System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()