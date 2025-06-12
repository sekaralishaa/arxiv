import streamlit as st
import requests

st.title("🔍 Scientific Article Recommender (Client)")

title = st.text_input("📝 Judul Artikel")
keywords = st.text_input("🔑 Keyword (pisahkan dengan koma)")
category = st.text_input("📂 Kategori Utama (misal: computer science)")

if st.button("🔎 Cari Rekomendasi"):
    if not any([title, keywords, category]):
        st.warning("⚠️ Masukkan minimal satu input.")
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
