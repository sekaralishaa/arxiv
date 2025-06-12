
import streamlit as st
import requests

# ======= Global Config ========
st.set_page_config(page_title="Scientific Article Recommendation System", layout="centered")

# ======= Custom Style ========
st.markdown("""
    <style>
    html, body {
        background-color: #FAF9F6;
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
        width: 100%;
        height: 60px;
        background-color: #B25640;
        margin: 0 auto 2rem auto;
    }
    .button-style button {
        background-color: #B25640 !important;
        color: white !important;
        border-radius: 20px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ======= Layout ========
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
title = st.text_input("Misal: scientific paper recommender system using deep learning and link prediction in citation network", key="title")

st.markdown("### 🔑 Keyword (pisahkan dengan koma)")
keywords = st.text_input("Misal: recommendation system, text processing, content based recommendation system", key="keywords")

st.markdown("### 📂 Kategori Utama")
category = st.text_input("Misal: computer science, mathematics", key="category")

# ======= Submit Button ========
with st.container():
    st.markdown('<div style="text-align: center;" class="button-style">', unsafe_allow_html=True)
    if st.button("Submit"):
        if not any([title, keywords, category]):
            st.warning("⚠️ Harap isi setidaknya satu input.")
        else:
            payload = {
                "title": title,
                "keywords": keywords,
                "category": category
            }
            try:
                with st.spinner("⏳ Mengirim ke server..."):
                    response = requests.post("http://localhost:5000/recommend", json=payload)
                    
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

st.markdown('<div class="rectangle"></div>', unsafe_allow_html=True)
