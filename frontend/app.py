import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Scientific Article Recommendation System", layout="centered")

# CSS Custom Styling
st.markdown("""
    <style>
    html, body, .main {
        background-color: #F8F7F0 !important;
        font-family: 'Arial', sans-serif;
        color: #222B52;
    }
    .header-banner {
        width: 100%;
        height: 80px;
        background: linear-gradient(to right, #B25640, #DE805C);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 26px;
        font-weight: bold;
        margin-bottom: 2rem;
        border-radius: 8px;
        letter-spacing: 1px;
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
    .submit-container {
        display: flex;
        justify-content: center;
        margin-top: 2rem;
        margin-bottom: 6rem;
    }
    .stButton > button {
        background-color: #B25640;
        color: #F8F7F0;
        font-weight: bold;
        padding: 0.6rem 2rem;
        border-radius: 25px;
        border: none;
        margin-top: 1rem;
    }
    input[type="text"] {
        background-color: transparent !important;
        border: 1.5px solid #222B52 !important;
        border-radius: 10px !important;
        padding: 8px !important;
        color: #222B52 !important; 
    }

    /* ✅ Tambahan untuk tabel hasil */
    thead tr th {
        color: #222B52 !important;
        font-weight: bold !important;
        border: 1px solid #222B52 !important;
        background-color: #F8F7F0 !important;
    }
    tbody tr td {
        color: #222B52 !important;
        border: 1px solid #222B52 !important;
        background-color: #F8F7F0 !important;
    }
    .stDataFrame {
        background-color: #F8F7F0 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Header banner
st.markdown('<div class="header-banner">Scientific Article Recommendation System</div>', unsafe_allow_html=True)

# Penjelasan sistem
st.markdown(
    """
<div class="description">
Aplikasi ini bertujuan untuk membantu pengguna menemukan artikel ilmiah yang relevan berdasarkan <b>judul artikel</b>, <b>keyword</b>, dan <b>kategori utama</b> yang dimasukkan oleh pengguna. 
Sistem ini sangat bermanfaat bagi mahasiswa, peneliti, dan praktisi untuk mempercepat proses pencarian referensi berkualitas dalam berbagai bidang sains dan teknologi.
<br><br>
Proses rekomendasi dilakukan dengan menggunakan pendekatan <b>Content-Based Filtering</b>, di mana sistem menghitung kemiripan antara input pengguna dan kumpulan artikel ilmiah dari basis data ArXiv. 
Fitur teks diproses menggunakan teknik <b>TF-IDF dan Word2Vec</b> untuk merepresentasikan konten secara numerik.
<br><br>
<b>Instruksi:</b>
<ol>
    <li>Masukkan judul artikel yang ingin kamu cari. Judul lengkap akan memberikan hasil lebih akurat, tetapi kamu juga bisa memasukkan bagian dari judul atau kata kunci utama.</li>
    <li>Masukkan keyword yang relevan, dipisahkan dengan koma. Misalnya: <i>recommendation system, machine learning</i>.</li>
    <li>Masukkan kategori utama artikel, seperti: <i>computer science</i>, <i>mathematics</i>, atau lainnya sesuai bidang.</li>
    <li>Klik tombol <b>Submit</b>, dan sistem akan menampilkan artikel yang paling relevan berdasarkan input kamu.</li>
</ol>
</div>
""",
    unsafe_allow_html=True
)



# Rules
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
st.markdown("#### 📝 Judul Artikel")
title = st.text_input("Masukkan judul lengkap atau kata kunci utama dari artikel")

st.markdown("#### 🔑 Keyword (pisahkan dengan koma)")
keywords = st.text_input("Contoh: recommendation system, text processing, machine learning")

st.markdown("#### 📂 Kategori Utama")
category = st.selectbox(
    "Pilih kategori utama artikel",
    [
        "physics",
        "mathematics",
        "computer science",
        "quantitative biology",
        "quantitative finance",
        "statistics",
        "electrical engineering and systems science",
        "economics"
    ]
)


# Submit button
col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    submit_clicked = st.button("Submit")

# Recommendation processing
if submit_clicked:
    if not any([title, keywords, category]):
        st.warning("⚠️ Harap isi setidaknya satu input.")
    else:
        try:
            with st.spinner("⏳ Mengirim ke server... mohon menunggu, kira-kira hasil akan keluar setelah 1 menit 😁"):
                response = requests.post("http://localhost:5000/recommend", json={
                    "title": title,
                    "keywords": keywords,
                    "category": category
                })
            if response.ok:
                result = response.json()
                df_result = pd.DataFrame(result)[["title", "authors", "categories", "abstract", "doi", "similarity_score"]]
                st.success("✅ Rekomendasi ditemukan!")
                st.dataframe(df_result)
            else:
                st.error(f"❌ Gagal mengambil rekomendasi (status {response.status_code})")
                st.text(response.text)
        except Exception as e:
            st.error(f"❌ Error: {e}")

# Bottom rectangle
st.markdown('<div class="rectangle"></div>', unsafe_allow_html=True)
