import cv2
from deepface import DeepFace
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

# Load face detector model
net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

# Start video capture
cap = cv2.VideoCapture(0)

# Set the resolution to 640x480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 420)

# Create a Tkinter window
window = tk.Tk()

# Create a label in the window to display video feed
label = tk.Label(window)
label.pack()

# Create labels to display detection results
gender_label = tk.Label(window)
gender_label.pack()
age_label = tk.Label(window)
age_label.pack()
emotion_label = tk.Label(window)
emotion_label.pack()

def update_image():
    # Read frame
    ret, frame = cap.read()

    if not ret:
        return

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

            # Update detection result labels
            gender_label.config(text=f"Gender: {gender}")
            age_label.config(text=f"Age: {age}")
            emotion_label.config(text=f"Emotion: {emotion}")

            # Draw rectangle around the face
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # Draw gender on the frame
            cv2.putText(frame, f"{gender}, {age}, {emotion}", (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which Tkinter uses)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    photo = ImageTk.PhotoImage(image=Image.fromarray(frame))

    # Update the image displayed on the label
    label.config(image=photo)
    label.image = photo

    # Call this function again after 10 milliseconds
    window.after(10, update_image)

# Call the update_image function to start
update_image()

# Start the Tkinter main loop
window.mainloop()

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()