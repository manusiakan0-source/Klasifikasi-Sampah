import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import json

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Klasifikasi Sampah",
    page_icon="♻️",
    layout="centered"
)

st.title("♻️ Aplikasi Klasifikasi Sampah")
st.write("Unggah foto sampah untuk mendeteksi jenis dan cara pengelolaannya secara otomatis.")

# Kamus Informasi Sampah
INFO = {
    "battery":     ("Baterai", "B3 (berbahaya)", "Serahkan ke drop point limbah B3/elektronik."),
    "biological":  ("Sampah Organik", "Organik", "Bisa dikomposkan atau dijadikan pupuk."),
    "brown-glass": ("Kaca Coklat", "Anorganik", "Bilas, lalu daur ulang sebagai kaca."),
    "cardboard":   ("Kardus", "Anorganik", "Lipat, jaga tetap kering, setor ke bank sampah."),
    "clothes":     ("Pakaian", "Tekstil", "Donasikan jika layak pakai."),
    "green-glass": ("Kaca Hijau", "Anorganik", "Bilas, lalu daur ulang sebagai kaca."),
    "metal":       ("Logam", "Anorganik", "Bersihkan, setor ke pengepul logam."),
    "paper":       ("Kertas", "Anorganik", "Jaga tetap kering, lalu daur ulang."),
    "plastic":     ("Plastik", "Anorganik", "Bilas dan daur ulang sesuai jenisnya."),
    "shoes":       ("Sepatu", "Tekstil/Anorganik", "Donasikan jika masih layak."),
    "trash":       ("Residu", "Residu", "Buang ke tempat sampah residu."),
    "white-glass": ("Kaca Bening", "Anorganik", "Bilas, lalu daur ulang sebagai kaca."),
}

IMG = (224, 224)

# Load Model dan Class Names dari file di GitHub
@st.cache_resource
def load_model_and_classes():
    model = keras.models.load_model("garbage_model.keras")
    with open("class_names.json", "r") as f:
        class_names = json.load(f)
    return model, class_names

with st.spinner("Memuat model AI, mohon tunggu..."):
    try:
        model, class_names = load_model_and_classes()
        st.success("Model berhasil dimuat!")
    except Exception as e:
        st.error(f"Gagal memuat model: Pastikan file 'garbage_model.keras' & 'class_names.json' sudah di-upload. Error: {e}")
        st.stop()

# Widget Upload Foto Sampah
uploaded_file = st.file_uploader("Foto sampah", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diunggah", use_container_width=True)
    
    if st.button("Cek Jenis Sampah", type="primary"):
        with st.spinner("Sedang memproses..."):
            img = image.resize(IMG)
            img_array = np.array(img, dtype="float32")
            probs = model.predict(np.expand_dims(img_array, 0), verbose=0)[0]
            
            scores = {INFO[class_names[i]][0]: float(probs[i]) for i in range(len(class_names))}
            k = int(np.argmax(probs))
            nama, kategori, tips = INFO[class_names[k]]
            
            st.markdown(f"### {nama}")
            st.markdown(f"**Kategori:** {kategori}")
            st.info(f"💡 **Tips Pengelolaan:** {tips}")
            
            if probs[k] < 0.6:
                st.warning("⚠️ Model kurang yakin, coba foto lain dengan latar polos.")
                
            st.markdown("---")
            st.subheader("Skor Prediksi:")
            st.bar_chart(scores)
          
