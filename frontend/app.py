import streamlit as st
import requests

# ======= Config Layout ========
st.set_page_config(page_title="Scientific Article Recommendation System", layout="centered")

# ======= Custom CSS ========
st.markdown("""
    <style>
    html, body {
        background-color: #F8F7F0;
    }
    .main {
        font-family: 'Arial';
    }
    .title {
        color: #222B52;
        font-size: 36px;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        color: #222B52;
        font-size: 24px;
        font-weight: 600;
    }
    .description {
        color: #222B52;
        font-size: 16px;
        margin-bottom: 1rem;
    }
    .rectangle {
        width: 100vw;
        height: 60px;
        background-color: #B25640;
        margin-bottom: 2rem;
    }
    .stTextInput > div > input {
        background-color: #F8F7F0 !important;
        color: #222B52 !important;
    }
    .custom-button button {
        background-color: #B25640 !important;
        color: #F8F7F0 !important;
        border-radius: 20px !important;
        padding: 0.5rem 1.5rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ======= UI ========
st.markdown('<div class="rectangle"></div>', unsafe_allow_html=True)
st.markdown('<div class="title">Scientific Paper Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Aturan</div>', unsafe_allow_html=True)

st.markdown("""
<div class="description">
    <ol>
        <li>Mohon masukkan input dengan bahasa Inggris dan pastikan input benar serta tidak ada typo.</li>
        <li>Sistem rekomendasi ini hanya menyajikan artikel dalam bidang <b>Fisika</b>, <b>Matematika</b>, <b>Ilmu Komputer</b>, <b>Biologi Kuantitatif</b>, <b>Keuangan Kuantitatif</b>, <b>Statistika</b>, <b>Teknik Elektro dan Ilmu Sistem</b>, dan <b>Ekonomi</b>.</li>
    </ol>
</div>
""", unsafe_allow_html=True)

# ======= Input Fields ========
st.markdown("### 📝 Judul Artikel")
title = st.text_input("Misal: scientific paper recommender system using deep learning and link prediction in citation network")

st.markdown("### 🔑 Keyword (pisahkan dengan koma)")
keywords = st.text_input("Misal: recommendation system, text processing, content based recommendation system")

st.markdown("### 📂 Kategori Utama")
category = st.text_input("Misal: computer science, mathematics")

# ======= Submit Button ========
st.markdown('<div class="custom-button" style="text-align: center;">', unsafe_allow_html=True)
if st.button("Submit"):
    if not any([title, keywords, category]):
        st.warning("⚠️ Harap isi setidaknya satu input.")
    else:
        try:
            with st.spinner("⏳ Mengirim ke server..."):
                response = requests.post("http://localhost:5000/recommend", json={
                    "title": title,
                    "keywords": keywords,
                    "category": category
                })
            if response.ok:
                result = response.json()
                st.success("✅ Rekomendasi ditemukan!")
                st.dataframe(result)
                st.caption(f"Status: {response.status_code}")
            else:
                st.error(f"❌ Gagal mengambil rekomendasi (status {response.status_code})")
                st.text(response.text)
        except Exception as e:
            st.error(f"❌ Error: {e}")
st.markdown('</div>', unsafe_allow_html=True)

# ======= Bottom Rectangle ========
st.markdown('<div class="rectangle"></div>', unsafe_allow_html=True)
