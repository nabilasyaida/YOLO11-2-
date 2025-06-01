import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tempfile
import os
from ultralytics import YOLO

# Load model YOLO
model = YOLO("best.pt")  # Ganti dengan path model kamu

# Background styling
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("Hologram.png");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Acne Detection Web App 💖")
st.write("Upload an image, video, or use your webcam to detect acne types!")

# Input options
option = st.radio("Choose input type:", ("Image", "Video", "Webcam"))

# Rekomendasi skincare
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

def classify_acne(count):
    if count > 10:
        return "Severe"
    elif count == 10:
        return "Mild"
    else:
        return "Normal"

# ====== IMAGE HANDLING ======
if option == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        image_np = np.array(image)

        results = model.predict(image_np, imgsz=640)
        result_img = results[0].plot()
        st.image(result_img, caption="Detected Acne", use_container_width=True)

        detected_classes = [results[0].names[int(cls)] for cls in results[0].boxes.cls.cpu().numpy()]
        display_recommendations(detected_classes)

        acne_count = len(detected_classes)
        st.subheader("📊 Acne Count and Severity Classification")
        st.write(f"Jumlah jerawat terdeteksi: **{acne_count}**")
        st.markdown(f"**Klasifikasi Tingkat Jerawat:** `{classify_acne(acne_count)}`")

# ====== VIDEO HANDLING ======
elif option == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        detected_classes_all = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, imgsz=640)
            result_frame = results[0].plot()

            stframe.image(result_frame, channels="BGR", use_container_width=True)

            detected_classes_frame = [results[0].names[int(cls)] for cls in results[0].boxes.cls.cpu().numpy()]
            detected_classes_all.extend(detected_classes_frame)

        cap.release()
        os.remove(tfile.name)

        acne_count = len(detected_classes_all)
        st.subheader("📊 Acne Count and Severity Classification")
        st.write(f"Total jerawat terdeteksi di video: **{acne_count}**")
        st.markdown(f"**Klasifikasi Tingkat Jerawat:** `{classify_acne(acne_count)}`")
        display_recommendations(detected_classes_all)

# ====== WEBCAM HANDLING ======
elif option == "Webcam":
    st.write("📷 Turn on your webcam for live acne detection")
    run = st.checkbox('Start Webcam')

    FRAME_WINDOW = st.image([])
    detected_classes_all = []

    cap = cv2.VideoCapture(0)  # Webcam default

    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("Gagal mengakses webcam")
            break

        results = model.predict(frame, imgsz=640)
        result_frame = results[0].plot()

        FRAME_WINDOW.image(result_frame, channels="BGR", use_container_width=True)

        detected_classes_frame = [results[0].names[int(cls)] for cls in results[0].boxes.cls.cpu().numpy()]
        detected_classes_all.extend(detected_classes_frame)

    else:
        cap.release()
        cv2.destroyAllWindows()

        st.subheader("📊 Hasil Deteksi Webcam")
        acne_count = len(detected_classes_all)
        st.write(f"Jumlah jerawat terdeteksi selama sesi: **{acne_count}**")
        st.markdown(f"**Klasifikasi Tingkat Jerawat:** `{classify_acne(acne_count)}`")
        display_recommendations(detected_classes_all)
