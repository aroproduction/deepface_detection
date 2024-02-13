import cv2
import streamlit as st
import numpy as np
from PIL import Image
from deepface import DeepFace

st.title("Real Time Face Detection")

# Load the cascade
net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

# Add a slider to the sidebar
FONT_SIZE = st.sidebar.slider('Font size', min_value=0.1, max_value=3.0, value=0.5, step=0.1)
FONT_WEIGHT = st.sidebar.slider('Font weight', min_value=0.1, max_value=3.0, value=0.5, step=0.1)
BORDER_WIDTH = st.sidebar.slider('Border width', min_value=0.1, max_value=3.0, value=0.5, step=0.1)

# Function for detecting faces
def detect_faces(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    face_count = 0  # Initialize face counter

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
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), int(BORDER_WIDTH))

            # Draw gender on the frame
            cv2.putText(frame, f"{gender}, {age}, {emotion}", (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0,255,0), int(FONT_WEIGHT))

            face_count += 1  # Increment face counter

    return frame, face_count

# Upload the image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)

    # Detect and display faces
    result_image, face_count = detect_faces(image)
    st.image(result_image, caption='Uploaded Image.', use_column_width=True)
    st.write("Number of faces detected: ", face_count)