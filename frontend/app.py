import streamlit as st
import requests

st.title("🔍 Scientific Article Recommender (Client)")

title = st.text_input("Judul")
keywords = st.text_input("Keyword")
category = st.text_input("Kategori")

if st.button("Cari Rekomendasi"):
    input_text = f"{title} {keywords.replace(',', ' ')} {category}"
    try:
        with st.spinner("⏳ Mengirim ke server..."):
            response = requests.post("http://31.97.187.177:5000/recommend", json={"query": input_text})
            
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
