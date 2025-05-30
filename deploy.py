import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tempfile
import os
from ultralytics import YOLO

# Load model
model = YOLO("yolov11-best.pt")  # Ganti dengan model acne kamu

# Background styling
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("assets/pastel_bg.png");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Acne Detection Web App 💖")
st.write("Upload an image or video to detect acne types!")

option = st.radio("Choose input type:", ("Image", "Video"))

if option == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        image_np = np.array(image)

        results = model.predict(image_np, imgsz=640)
        result_img = results[0].plot()
        st.image(result_img, caption="Detected Acne", use_column_width=True)

elif option == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(frame, imgsz=640)
            result_frame = results[0].plot()
            stframe.image(result_frame, channels="BGR", use_column_width=True)

        cap.release()
        os.unlink(tfile.name)
