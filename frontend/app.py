import streamlit as st
import requests

# Page config
st.set_page_config(page_title="Scientific Article Recommendation System", layout="centered")

# CSS Styling
st.markdown("""
    <style>
    html, body, .main {
        background-color: #F8F7F0 !important;
        font-family: 'Arial', sans-serif;
        color: #222B52;
    }
    .rectangle {
        width: 100vw;
        height: 60px;
        background-color: #B25640;
        position: relative;
        left: -8vw;
    }
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #222B52;
        margin-bottom: 0.3rem;
    }
    .subtitle {
        font-size: 22px;
        font-weight: 600;
        color: #222B52;
        margin-top: 1.2rem;
    }
    .description {
        font-size: 15px;
        color: #222B52;
        margin-bottom: 1rem;
    }
    .stTextInput > div > input {
        background-color: #F8F7F0 !important;
        color: #222B52 !important;
    }
    .submit-container {
        display: flex;
        justify-content: center;
        margin-top: 2rem;
        margin-bottom: 5rem;
    }
    .stButton > button {
        background-color: #B25640;
        color: #F8F7F0;
        font-weight: bold;
        padding: 0.5rem 2rem;
        border-radius: 20px;
        border: none;
    }
    </style>
""", unsafe_allow_html=True)

# Top rectangle
st.markdown('<div class="rectangle"></div>', unsafe_allow_html=True)

# Header and rules
st.markdown('<div class="title">Scientific Paper Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Rules</div>', unsafe_allow_html=True)
st.markdown("""
<div class="description">
    <ol>
        <li>Mohon masukkan input dengan bahasa Inggris dan pastikan input benar serta tidak ada typo.</li>
        <li>Sistem rekomendasi ini hanya menyajikan artikel dalam bidang <b>Fisika</b>, <b>Matematika</b>, <b>Ilmu Komputer</b>, <b>Biologi Kuantitatif</b>, <b>Keuangan Kuantitatif</b>, <b>Statistika</b>, <b>Teknik Elektro dan Ilmu Sistem</b>, dan <b>Ekonomi</b>.</li>
    </ol>
</div>
""", unsafe_allow_html=True)

# Input fields
st.markdown("### 📝 Judul Artikel")
title = st.text_input("Misal: scientific paper recommender system using deep learning and link prediction in citation network")

st.markdown("### 🔑 Keyword (pisahkan dengan koma)")
keywords = st.text_input("Misal: recommendation system, text processing, content based recommendation system")

st.markdown("### 📂 Kategori Utama")
category = st.text_input("Misal: computer science")

# Submit button center-aligned
st.markdown('<div class="submit-container">', unsafe_allow_html=True)
submit_clicked = st.button("Submit")
st.markdown('</div>', unsafe_allow_html=True)

# Logic
if submit_clicked:
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

# Bottom rectangle
st.markdown('<div class="rectangle"></div>', unsafe_allow_html=True)
