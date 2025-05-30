from ultralytics import YOLO
import streamlit as st
from collections import Counter
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

st.set_page_config(page_title="YOLO Acne Detection", layout="centered")
st.title("YOLO Acne Detection")

# Load model
model_path = "best.pt"
if not os.path.exists(model_path):
    st.error("Model 'best.pt' tidak ditemukan. Silakan unggah model.")
    st.stop()
else:
    model = YOLO(model_path)

# Upload gambar
uploaded_file = st.file_uploader("Unggah gambar wajah dengan jerawat", type=["jpg", "jpeg", "png"])
if uploaded_file is None:
    st.warning("Silakan unggah gambar terlebih dahulu.")
    st.stop()

# Buka gambar dari upload
image = Image.open(uploaded_file).convert("RGB")
image_np = np.array(image)

# Jalankan deteksi
results = model(image_np)[0]
class_ids = results.boxes.cls.cpu().numpy()
class_names = results.names
counts = Counter(class_ids)

# Gambar hasil deteksi
img_with_boxes = results.plot()
img_pil = Image.fromarray(img_with_boxes)

# Tambahkan background (opsional)
bg_path = "Pastel Pink Holographic Gradient Mouse Pad Background.png"
if os.path.exists(bg_path):
    background = Image.open(bg_path).convert("RGB")
    background = background.resize(img_pil.size)
    background.paste(img_pil, (0, 0), img_pil if img_pil.mode == 'RGBA' else None)
else:
    background = img_pil
    st.warning("Background tidak ditemukan, hanya menampilkan hasil deteksi.")

# Tambahkan label jumlah jerawat
draw = ImageDraw.Draw(background)
try:
    font = ImageFont.truetype("arial.ttf", 24)
except:
    font = ImageFont.load_default()

y_offset = 30
for i, (class_id, count) in enumerate(counts.items()):
    name = class_names[int(class_id)]
    label = f"{name}: {count}"
    draw.text((10, y_offset + i * 30), label, font=font, fill=(255, 0, 0))

# Tampilkan hasil akhir
st.image(background, caption="Hasil Deteksi Jerawat dengan YOLO", use_column_width=True)
