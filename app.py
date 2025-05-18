# ===================================
# 3. app.py (file utama untuk Streamlit)
# ===================================
import streamlit as st
import pandas as pd
import numpy as np
import gdown
import os
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

@st.cache_resource
def download_and_load_data():
    import os
    import gdown
    import pandas as pd
    from scipy import sparse
    from gensim.models import KeyedVectors

    # Buat folder data
    os.makedirs("data", exist_ok=True)

    # --- GDrive URLs ---
    parquet_url = "https://drive.google.com/uc?id=1l56SscV4A8yHuJ0Zi2OzEu8hY4E6-fGA"
    word2vec_url = "https://drive.google.com/uc?id=1VIqC0of1XGTTQiAKVThEyzg9vsTOzGg0"

    parquet_path = "data/df_final.parquet"
    w2v_path = "data/GoogleNews-vectors-negative3001.bin"

    # Download parquet
    if not os.path.exists(parquet_path):
        gdown.download(parquet_url, parquet_path, quiet=False)

    # Download Word2Vec
    if not os.path.exists(w2v_path):
        gdown.download(word2vec_url, w2v_path, quiet=False)

    # --- Load ---
    df = pd.read_parquet(parquet_path)
    w2v_model = KeyedVectors.load_word2vec_format(w2v_path, binary=True)

    # --- Load Word2Vec Chunks ---
    chunk_ids = [
        "19cn1jpODxFgv1VT6enXI-gF3t4pDTteE",  # word2vec_chunk_hybrid_01.npz 
        "1Bbbbbbb5678yyyy",  # word2vec_chunk_hybrid_02.npz
        # dst... tambahkan ID semua file
    ]

    chunks = []
    for i, chunk_id in enumerate(chunk_ids, start=1):
        chunk_path = f"data/word2vec_chunk_hybrid_{i:02d}.npz"
        chunk_url = f"https://drive.google.com/uc?id={chunk_id}"

        if not os.path.exists(chunk_path):
            gdown.download(chunk_url, chunk_path, quiet=False)

        chunks.append(sparse.load_npz(chunk_path).toarray())

    return df, w2v_model, chunks

def get_word2vec_vector(text, model):
    words = text.lower().split()
    vectors = [model[word] for word in words if word in model]
    if not vectors:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(vectors, axis=0).astype(np.float32)


def recommend(df, model, chunks, user_text):
    query_vec = get_word2vec_vector(user_text, model).reshape(1, -1)

    all_scores = []
    all_indexes = []
    start_idx = 0
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
    with st.spinner("🔄 Memuat data dan memproses rekomendasi..."):
        df_final, word2vec_model, w2v_chunks = download_and_load_data()
        combined_input = f"{user_title} {user_keywords.replace(',', ' ')} {user_category}"
        top_10 = recommend(df_final, word2vec_model, w2v_chunks, combined_input)

    st.success("✅ Rekomendasi ditemukan!")
    st.dataframe(top_10[['title', 'authors', 'categories_clean', 'similarity_score']])
