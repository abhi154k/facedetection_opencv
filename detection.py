import cv2
import streamlit as st
import numpy as np
import os
from datetime import datetime
import time

# Load pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def save_image(frame):
    if not os.path.exists('captured_faces'):
        os.makedirs('captured_faces')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'captured_faces/face_{timestamp}.jpg'
    cv2.imwrite(filename, frame)
    st.write(f"Image saved: {filename}")

st.title("Real-time Face Detection")

run = st.checkbox('Run Face Detection')

FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

if run:
    stop_button = st.button('Stop')  # Create the stop button outside the loop
    last_captured = time.time()  # Track the time of the last capture

    while True:
        ret, frame = camera.read()
        if not ret:
            st.error("Failed to capture image from camera")
            break

        faces = detect_faces(frame)
        current_time = time.time()

        if len(faces) > 0 and (current_time - last_captured) > 2:
            save_image(frame)
            last_captured = current_time  # Update the last captured time
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

        if stop_button:
            break

camera.release()
