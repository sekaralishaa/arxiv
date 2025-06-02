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
            response = requests.post("https://arxiv-backend.onrender.com/recommend", json={"query": input_text})
            st.write("Status Code:", response.status_code)
            st.write("Response Text:", response.text)
            result = response.json()
            st.success("✅ Rekomendasi ditemukan!")
            st.dataframe(result)
    except Exception as e:
        st.error(f"❌ Error: {e}")
