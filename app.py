# app.py
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
    # === Konfigurasi path & link ===
    os.makedirs("data", exist_ok=True)

    parquet_id = "1l56SscV4A8yHuJ0Zi2OzEu8hY4E6-fGA"  # ID Google Drive untuk df_final.parquet
    parquet_path = "data/df_final.parquet"
    w2v_path = "data/GoogleNews-vectors-reduced.bin"

    # === Download df_final.parquet dari Google Drive ===
    if not os.path.exists(parquet_path):
        st.write("📥 Downloading df_final.parquet from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={parquet_id}", parquet_path, quiet=False)

    # === Cek keberadaan Word2Vec bin file (disimpan di repo via Git LFS) ===
    if not os.path.exists(w2v_path):
        st.error("❌ File Word2Vec model (GoogleNews-vectors-reduced.bin) tidak ditemukan di repo.")
        raise FileNotFoundError("Word2Vec .bin model not found in 'data/'")

    # === Load DataFrame dan Model ===
    df = pd.read_parquet(parquet_path)
    w2v_model = KeyedVectors.load_word2vec_format(w2v_path, binary=True)

    # === Load Word2Vec Chunks dari Google Drive ===
    chunk_ids = [
        "19cn1jpODxFgv1VT6enXI-gF3t4pDTteE", "17Vhxw4ErpRDE_rDZYQHNT8BqU0VuPeRM", "14pzNVmmuMTOfUy19gPeGWv-dsOy-Ygbn",
        "1VYoGMF5Qv2E9Bn20mRKM-pV61t1j58Pi", "1Qnr9IHH8LvnpTtBU2xsfmsTi5KE7nxus", "1AFb7dX8Dl1qWyYcYXnkq38PQJtx8nR5E",
        "1z7CJAqkEToG9uOXQVIZ7Kl7MdVFVSlto", "1rYVsRFb8lNeS0bpxTOuTDwOyYD1jSEeg", "135l1x0JTbOIM3f84CWHRh99pNj0S5VC_",
        "14RxEsGkkfuHxWyQ-K5ufwMzmTHHyBBdL", "1o0Mhjms9TT2t1N5AOF4FIyG0XNTn-GKY", "18rC3qMOVpW0lSf0jvhRng0cS40zRlBAi",
        "1SCBlEqcQmlnQ4Xh0PFfKchhj4wQHo0gD", "1WdKXBsq8hEX7i2HxueYMK1IOqgXYI2TB", "1YhBWBm5gVmOTKncrsX4mWEC4gDYE21q-",
        "1NhIxabx9Y73JgPix9XWA5EYLOAoPsSy4", "1Vb-KaCrRi5nODkadlQ_4HyMdkQ_zGQST", "1p78fcR7977lV32Hsnhq4bHmV-g1GGfvu",
        "1GkcP3HF5idKmKuR9qJoRQ6RXx9V-khUE", "1kDzJ4DLkJhpWUFf0DT4lcmwLSDG6GNlW", "1VZMw2_v4mLEZsG30Ni8Kfqc9dfNcaYFN",
        "1P_mACznvdZA4d8iwxSxSPksTwZLlRr13", "1FGlP7qW2KbOEAAf8ywooIyrNx4dPEMlj", "1XGvFFusoGFX1hi9XqmlQkCwHTMV0qbI3",
        "12LDvIZwZic_BtyeqDdNcaeuf863VSAvm", "1ivOI8mUAvHwL5ryDkxxYdG_nsSmcleW_", "1poJngaPThtWy__NauBZNiUQ5TMLaMNWr"
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
    return np.mean(vectors, axis=0).astype(np.float32) if vectors else np.zeros(model.vector_size)

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

# === UI ===
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
