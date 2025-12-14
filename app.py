import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(page_title="Cat or Dog Classifier", layout="centered")
st.title("Image Classification: Cat or Dog")

MODEL_PATH = r"C:/Users/wisam/Documents/ImageClasify_CatDog/Image_classify.keras"
model = load_model(MODEL_PATH)

CLASS_NAMES = ['cats', 'dogs']
IMG_HEIGHT = 224
IMG_WIDTH = 224

DEFAULT_DIR = "default_images"
os.makedirs(DEFAULT_DIR, exist_ok=True)

default_files = sorted([f for f in os.listdir(DEFAULT_DIR)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

UPLOAD_OPTION = "-- Upload (pilih file) --"
options = [UPLOAD_OPTION] + default_files
selected = st.selectbox("Pilih gambar default atau pilih opsi untuk upload:", options)

uploaded_file = None
if selected == UPLOAD_OPTION:
    uploaded_file = st.file_uploader("Upload gambar (jpg/png)", type=["jpg", "jpeg", "png"])
else:
    image_path = os.path.join(DEFAULT_DIR, selected)
    if os.path.exists(image_path):
        st.image(image_path, caption=f"Default: {selected}", width=300)
    else:
        st.warning(f"Gambar default tidak ditemukan: {image_path}")

image_load = None
if uploaded_file is not None:
    try:
        image_load = Image.open(uploaded_file).convert("RGB")
        st.image(image_load, caption="Gambar yang diupload", width=300)
    except Exception as e:
        st.error("Gagal membaca file yang diupload: " + str(e))
elif selected != UPLOAD_OPTION and os.path.exists(os.path.join(DEFAULT_DIR, selected)):
    image_load = Image.open(os.path.join(DEFAULT_DIR, selected)).convert("RGB")

if image_load is not None:
    image_resized = image_load.resize((IMG_WIDTH, IMG_HEIGHT))
    img_arr = tf.keras.utils.img_to_array(image_resized)
    img_bat = np.expand_dims(img_arr, axis=0)

    try:
        predict = model.predict(img_bat)
        score = tf.nn.softmax(predict[0])
        predicted_label = CLASS_NAMES[int(np.argmax(score))]
        acc = round(float(np.max(score) * 100), 2)

        st.markdown(f"**Hasil prediksi model:** `{predicted_label}`")
        st.markdown(f"**Akurasi:** `{acc}%`")

        probs = {CLASS_NAMES[i]: round(float(score[i] * 100), 2) for i in range(len(CLASS_NAMES))}
        st.write("Probabilitas tiap kelas (%):", probs)
    except Exception as e:
        st.error("Terjadi kesalahan saat melakukan prediksi: " + str(e))
else:
    if not default_files:
        st.info("Folder 'default_images'kosong. Silakan upload gambar untuk melakukan prediksi.")
    else:
        st.info("Pilih gambar dari atau upload gambar untuk melakukan prediksi.")
