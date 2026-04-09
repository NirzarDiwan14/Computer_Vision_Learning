import os
import numpy as np
import cv2
import pickle
from PIL import Image


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(
    "/home/nirzar-diwan/Desktop/computer_vision_learning/images/face_dataset"
)
print(IMAGE_DIR)

# Face cascades from OpenCV
face_cascade = cv2.CascadeClassifier("/home/nirzar-diwan/Desktop/computer_vision_learning/haar_cascades/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("/home/nirzar-diwan/Desktop/computer_vision_learning/haar_cascades/haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier("/home/nirzar-diwan/Desktop/computer_vision_learning/haar_cascades/haarcascade_smile.xml")


x_train = []
y_labels = []

currrent_id = 0
label_ids = {}

recognizer = cv2.face.LBPHFaceRecognizer()
print("Program started")
for root, dirs, files in os.walk(IMAGE_DIR):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()

            # print(label, ":", path)
            if not label in label_ids:
            
                label_ids[label] = currrent_id
                currrent_id += 1
            id_ = label_ids[label]
            # print(label_ids)

            pil_image = Image.open(path).convert("L")  # Convert Grayscale
            image_array = np.array(pil_image, "uint8")

            # finding ROI 
            faces = face_cascade.detectMultiScale(image_array, 1.5, 5)

            for (x, y, w, h) in faces:

                # Draw face
                roi = image_array[y:y + h, x:x + w]
                x_train.append(roi)
                y_labels.append(id_)
                # roi_color = frame[y:y + h, x:x + w]
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # cv2.putText(frame, "Face", (x, y - 5),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)


print("--------")
# print(y_labels)
# print(x_train)
print("--------")

with open("labels.pickle", "wb") as f:
    pickle.dump(label_ids,f)

# Training 

print("Training Started")
recognizer.train(x_train,np.array(y_labels))
print("Training done")
recognizer.save("trainer.yml")
print("Training Ended")

print("Program Ended")
