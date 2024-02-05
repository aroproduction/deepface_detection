import cv2
from deepface import DeepFace
import numpy as np

# Load face detector model
net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    # Read frame
    ret, frame = cap.read()

    if not ret:
        break

    # Detect faces
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # Loop over the detections
    for i in range(0, detections.shape[2]):
        # Extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
        if confidence > 0.5:
            # Compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure the detected bounding box does fall outside the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # Extract the face ROI and convert it from BGR to RGB channel
            face = frame[startY:endY, startX:endX]
            
            # Analyze face
            result = DeepFace.analyze(img_path = face, actions = ['gender', "emotion", "age"], enforce_detection=False)

            gender, age, emotion = result[0]['dominant_gender'], result[0]['age'], result[0]['dominant_emotion']

             # Draw rectangle around the face
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # Draw gender on the frame
            cv2.putText(frame, f"{gender}, {age}, {emotion}", (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()