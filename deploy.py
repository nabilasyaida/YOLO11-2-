import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os

# Load background
def set_bg(png_file):
    with open(png_file, "rb") as f:
        bg_data = f.read()
    bg_base64 = base64.b64encode(bg_data).decode()
    page_bg = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bg_base64}");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
    }}
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)

import base64
set_bg("background/Hologram.png")

st.title("💖 Acne Detection with YOLOv11")
st.write("Upload an image or video to detect acne types.")

# Load YOLO model
model = YOLO("model/best.pt")

# Input method
option = st.radio("Choose input type:", ("Image", "Video"))

if option == "Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Detecting..."):
            results = model.predict(np.array(image), imgsz=640)
            res_plotted = results[0].plot()
            st.image(res_plotted, caption="Detection Result", use_column_width=True)

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
