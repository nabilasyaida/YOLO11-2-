import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tempfile
import os
from ultralytics import YOLO

# Load model
model = YOLO("best.pt")  # Ganti dengan path modelmu

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
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        image_np = np.array(image)

        results = model.predict(image_np, imgsz=640)
        result_img = results[0].plot()
        st.image(result_img, caption="Detected Acne", use_column_width=True)
