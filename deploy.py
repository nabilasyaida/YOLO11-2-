import streamlit as st
from PIL import Image
import cv2
import tempfile
import os
from ultralytics import YOLO
import numpy as np

# Set background image
def set_background(png_file):
    with open(png_file, "rb") as file:
        bg_data = file.read()
    bg_base64 = base64.b64encode(bg_data).decode()
    bg_style = f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{bg_base64}");
            background-size: cover;
        }}
        </style>
    """
    st.markdown(bg_style, unsafe_allow_html=True)

# Load background
import base64
set_background("Pastel Pink Holographic Gradient Mouse Pad Background.png")

# Load YOLOv11 model
model = YOLO("best.pt")  # ganti dengan path ke model kamu

# Title
st.title("💖 Acne Detection Web App")

# Sidebar for image/video input
option = st.sidebar.selectbox("Choose Input Type", ["Image", "Video"])

if option == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # YOLO prediction
        results = model.predict(image)
        annotated = results[0].plot()  # get annotated image
        st.image(annotated, caption="Detected Acne", use_column_width=True)

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

            results = model.predict(frame)
            annotated_frame = results[0].plot()

            stframe.image(annotated_frame, channels="BGR")

        cap.release()
        os.remove(tfile.name)
