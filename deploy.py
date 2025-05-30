import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tempfile
import os
from ultralytics import YOLO

# Load model
model = YOLO("best.pt")  # Ganti dengan path model kamu

# Background styling
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("assets/Hologram.png");
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

# Dictionary rekomendasi skincare
skincare_recommendations = {
    "whitehead": "Gunakan pembersih berbasis salicylic acid dan hindari produk komedogenik.",
    "blackhead": "Eksfoliasi ringan 2-3 kali seminggu dengan BHA seperti salicylic acid.",
    "papule": "Gunakan toner atau serum dengan niacinamide dan hindari produk berbahan keras.",
    "pustule": "Gunakan spot treatment dengan benzoyl peroxide, dan pastikan wajah tetap bersih.",
    "nodule": "Konsultasi dengan dermatolog. Gunakan skincare yang non-irritant dan soothing.",
    "cyst": "Perlu penanganan medis. Jangan dipencet dan jaga kebersihan kulit wajah."
}

def display_recommendations(names_detected):
    st.subheader("🎯 Skincare Recommendations")
    shown = set()
    for cls in names_detected:
        if cls not in shown:
            rec = skincare_recommendations.get(cls, "Tidak ada rekomendasi khusus.")
            st.markdown(f"**{cls.capitalize()}**: {rec}")
            shown.add(cls)

if option == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        image_np = np.array(image)

        results = model.predict(image_np, imgsz=640)
        result_img = results[0].plot()
        st.image(result_img, caption="Detected Acne", use_container_width=True)

        # Deteksi kelas jerawat yang muncul
        detected_classes = [results[0].names[int(cls)] for cls in results[0].boxes.cls.cpu().numpy()]
        display_recommendations(detected_classes)
