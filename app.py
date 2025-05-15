import streamlit as st
import pandas as pd
import numpy as np
import gdown
import os
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from glob import glob
import re
import string

@st.cache_data
def load_dataset():
    if not os.path.exists("df_final.parquet"):
        gdown.download("https://drive.google.com/uc?id=YOUR_FILE_ID", "df_final.parquet", quiet=False)
    return pd.read_parquet("df_final.parquet")

@st.cache_resource
def load_model():
    return KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative3001.bin", binary=True)

@st.cache_resource
def load_vectors():
    files = sorted(glob("word2vec_chunk_hybrid_*.npz"))
    return [sparse.load_npz(f).toarray() for f in files]

def clean_user_input(text):
    if pd.isna(text):
        return ""
    text = re.sub(r"\$\([^)]*\)\$", "", text)
    text = text.replace("\n", " ").replace("-", " ")
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

def get_word2vec_vector(text, model):
    words = text.lower().split()
    vectors = [model[word] for word in words if word in model]
    if not vectors:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(vectors, axis=0).astype(np.float32)

def recommend_articles(user_title, user_keywords, user_category, df, model, chunks):
    raw = f"{user_title.strip()} {user_keywords.replace(',', ' ').strip()} {user_category.strip()}"
    cleaned = clean_user_input(raw)
    query_vec = get_word2vec_vector(cleaned, model).reshape(1, -1)

    all_scores = []
    all_indexes = []
    start_idx = 0

    for chunk in chunks:
        sim = cosine_similarity(query_vec, chunk).flatten()
        all_scores.extend(sim)
        all_indexes.extend(range(start_idx, start_idx + chunk.shape[0]))
        start_idx += chunk.shape[0]

    sim_df = df.iloc[all_indexes].copy()
    sim_df['similarity_score'] = all_scores
    return sim_df.sort_values(by='similarity_score', ascending=False).head(10)

# === Streamlit App ===
st.set_page_config(page_title="ArXiv Article Recommender", layout="wide")
st.title("📚 ArXiv Scientific Article Recommender")

with st.spinner("🔄 Loading data and model..."):
    df_final = load_dataset()
    word2vec_model = load_model()
    w2v_chunks = load_vectors()

st.markdown("### Masukkan input artikel untuk mendapatkan rekomendasi")
user_title = st.text_input("Judul artikel")
user_keywords = st.text_input("Keyword (pisahkan dengan koma)")
user_category = st.text_input("Kategori utama (misal: computer science)")

if st.button("🔍 Cari Artikel yang Relevan"):
    with st.spinner("🔍 Mencari artikel paling mirip..."):
        results = recommend_articles(
            user_title=user_title,
            user_keywords=user_keywords,
            user_category=user_category,
            df=df_final,
            model=word2vec_model,
            chunks=w2v_chunks
        )
        st.success("✅ Rekomendasi ditemukan!")
        st.dataframe(results[['title', 'authors', 'categories_clean', 'abstract', 'doi', 'similarity_score']])
