import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import base64
import os

# Set background
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Load model
model = YOLO("best.pt")  # pastikan model kamu ada di folder ini

# Set holographic background
set_background("Pastel Pink Holographic Gradient Mouse Pad Background.png")

# Title
st.markdown("<h1 style='text-align: center; color: white;'>💖 Acne Detection Web App</h1>", unsafe_allow_html=True)

# Sidebar
mode = st.sidebar.radio("Choose input type:", ["Image", "Video"])

if mode == "Image":
    img_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if img_file:
        img = Image.open(img_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Run detection
        results = model.predict(img)
        res_plotted = results[0].plot()

        st.image(res_plotted, caption="Detected Acne", use_column_width=True)

elif mode == "Video":
    vid_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    if vid_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(vid_file.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame)
            annotated = results[0].plot()
            stframe.image(annotated, channels="BGR", use_column_width=True)

        cap.release()
        os.remove(tfile.name)
