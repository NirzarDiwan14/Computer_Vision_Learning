from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# Optional: set resolution
cap.set(3, 1280)
cap.set(4, 720)

while True:
    success, frame = cap.read()
    if not success:
        break

    # Run YOLO prediction on frame
    results = model(frame)

    # Get annotated frame (with boxes, labels)
    annotated_frame = results[0].plot()

    # Show result
    cv2.imshow("YOLO Webcam", annotated_frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()