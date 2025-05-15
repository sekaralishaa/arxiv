# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
from scipy import sparse
from glob import glob
import re
import string

# === Load Data & Model ===
st.title("🔍 Sistem Rekomendasi Artikel Ilmiah (ArXiv)")

@st.cache_data

def load_data():
    df = pd.read_parquet("df_final.parquet")
    return df

@st.cache_resource

def load_model():
    model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative3001.bin", binary=True)
    return model

@st.cache_resource

def load_w2v_chunks():
    w2v_files = sorted(glob("word2vec_chunk_hybrid_*.npz"))
    return [sparse.load_npz(f).toarray() for f in w2v_files]

# === Utility ===
def clean_user_input(text):
    text = re.sub(r"\$\([^)]*\)\$", "", text)
    text = text.replace("\n", " ").replace("-", " ")
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

def get_word2vec_vector(text, model):
    words = text.lower().split()
    word_vectors = [model[word] for word in words if word in model]
    return np.mean(word_vectors, axis=0).astype(np.float32) if word_vectors else np.zeros(model.vector_size)

# === Recommendation Logic ===
def recommend_articles(title, keywords, category, df_final, model, w2v_chunks):
    raw_text = title.strip() + " " + keywords.replace(",", " ").strip() + " " + category.strip()
    query_text = clean_user_input(raw_text)
    query_vec = get_word2vec_vector(query_text, model).reshape(1, -1)

    all_scores = []
    all_indexes = []
    start_idx = 0

    for chunk in w2v_chunks:
        sims = cosine_similarity(query_vec, chunk).flatten()
        all_scores.extend(sims)
        all_indexes.extend(range(start_idx, start_idx + chunk.shape[0]))
        start_idx += chunk.shape[0]

    similarity_df = df_final.iloc[all_indexes].copy()
    similarity_df['similarity_score'] = all_scores
    top10 = similarity_df.sort_values(by='similarity_score', ascending=False).head(10)
    return top10[['title', 'authors', 'categories_clean', 'abstract', 'doi', 'similarity_score']]

# === UI ===
st.subheader("Masukkan Artikel yang Anda Cari")
title_input = st.text_input("Judul Artikel")
keywords_input = st.text_input("Keyword (pisahkan dengan koma)")
category_input = st.text_input("Kategori (misal: computer science)")

if st.button("Rekomendasikan"):
    with st.spinner("🔎 Mencari artikel terbaik untuk Anda..."):
        df_final = load_data()
        word2vec_model = load_model()
        w2v_chunks = load_w2v_chunks()

        results = recommend_articles(
            title_input,
            keywords_input,
            category_input,
            df_final,
            word2vec_model,
            w2v_chunks
        )

        st.success("Berikut adalah 10 artikel rekomendasi:")
        for i, row in results.iterrows():
            st.markdown(f"### {row['title']}")
            st.markdown(f"**Authors**: {row['authors']}")
            st.markdown(f"**Kategori**: {row['categories_clean']}")
            st.markdown(f"**DOI**: [{row['doi']}]({row['doi']})" if row['doi'] else "DOI: -")
            st.markdown(f"**Similarity Score**: {row['similarity_score']:.4f}")
            st.markdown(f"**Abstract**: {row['abstract']}")
            st.markdown("---")
