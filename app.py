# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

@st.cache_resource
def download_and_load_data():
    base_dir = "data/data"

    # ✅ Hanya ambil 2 chunk pertama untuk uji coba
    chunk_ids = [1, 2]

    # --- Load df_final chunks ---
    df_chunks = []
    for i in chunk_ids:
        chunk_path = os.path.join(base_dir, f"df_final_part_{i:02d}.parquet")
        if not os.path.exists(chunk_path):
            st.error(f"❌ File tidak ditemukan: {chunk_path}")
            raise FileNotFoundError(chunk_path)
        df_chunks.append(pd.read_parquet(chunk_path))

    df = pd.concat(df_chunks, ignore_index=True)

    # --- Load Word2Vec model (.bin) ---
    w2v_path = os.path.join(base_dir, "GoogleNews-vectors-reduced.bin")
    if not os.path.exists(w2v_path):
        st.error("❌ Word2Vec .bin model tidak ditemukan.")
        raise FileNotFoundError(w2v_path)
    w2v_model = KeyedVectors.load_word2vec_format(w2v_path, binary=True)

    # --- Load Word2Vec document vectors (.npz chunks) ---
    w2v_chunks = []
    for i in chunk_ids:
        npz_path = os.path.join(base_dir, f"word2vec_chunk_hybrid_{i:02d}.npz")
        if not os.path.exists(npz_path):
            st.error(f"❌ File tidak ditemukan: {npz_path}")
            raise FileNotFoundError(npz_path)
        w2v_chunks.append(sparse.load_npz(npz_path).toarray())

    return df, w2v_model, w2v_chunks


# @st.cache_resource
# def download_and_load_data():
#     base_dir = "data/data"

#     # --- Load df_final chunks ---
#     df_chunks = []
#     for i in range(1, 28):
#         chunk_path = os.path.join(base_dir, f"df_final_part_{i:02d}.parquet")
#         if not os.path.exists(chunk_path):
#             st.error(f"❌ File tidak ditemukan: {chunk_path}")
#             raise FileNotFoundError(chunk_path)
#         df_chunks.append(pd.read_parquet(chunk_path))

#     df = pd.concat(df_chunks, ignore_index=True)

#     # --- Load Word2Vec model (.bin) ---
#     w2v_path = os.path.join(base_dir, "GoogleNews-vectors-reduced.bin")
#     if not os.path.exists(w2v_path):
#         st.error("❌ Word2Vec .bin model tidak ditemukan.")
#         raise FileNotFoundError(w2v_path)
#     w2v_model = KeyedVectors.load_word2vec_format(w2v_path, binary=True)

#     # --- Load Word2Vec document vectors (.npz chunks) ---
#     w2v_chunks = []
#     for i in range(1, 28):
#         npz_path = os.path.join(base_dir, f"word2vec_chunk_hybrid_{i:02d}.npz")
#         if not os.path.exists(npz_path):
#             st.error(f"❌ File tidak ditemukan: {npz_path}")
#             raise FileNotFoundError(npz_path)
#         w2v_chunks.append(sparse.load_npz(npz_path).toarray())

#     return df, w2v_model, w2v_chunks


def get_word2vec_vector(text, model):
    words = text.lower().split()
    vectors = [model[word] for word in words if word in model]
    if not vectors:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(vectors, axis=0).astype(np.float32)


def recommend(df, model, chunks, user_text):
    query_vec = get_word2vec_vector(user_text, model).reshape(1, -1)
    all_scores, all_indexes, start_idx = [], [], 0

    for chunk in chunks:
        sims = cosine_similarity(query_vec, chunk).flatten()
        all_scores.extend(sims)
        all_indexes.extend(range(start_idx, start_idx + chunk.shape[0]))
        start_idx += chunk.shape[0]

    sim_df = df.iloc[all_indexes].copy()
    sim_df['similarity_score'] = all_scores
    return sim_df.sort_values(by='similarity_score', ascending=False).head(10)


# ======================
# Streamlit UI
# ======================
st.title("🔍 Scientific Article Recommendation System")

st.markdown("Masukkan informasi artikel yang Anda minati")
user_title = st.text_input("Judul Artikel")
user_keywords = st.text_input("Keyword (pisahkan dengan koma)")
user_category = st.text_input("Kategori utama (opsional)")

if st.button("Cari Rekomendasi"):
    try:
        with st.spinner("🔄 Memuat data dan memproses rekomendasi..."):
            df_final, word2vec_model, w2v_chunks = download_and_load_data()
            combined_input = f"{user_title} {user_keywords.replace(',', ' ')} {user_category}"
            top_10 = recommend(df_final, word2vec_model, w2v_chunks, combined_input)

        st.success("✅ Rekomendasi ditemukan!")
        st.dataframe(top_10[['title', 'authors', 'categories_clean', 'similarity_score']])
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan: {e}")
