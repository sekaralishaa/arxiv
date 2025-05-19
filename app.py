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
    word2vec_url = "https://drive.google.com/file/d/1yE7DmhFcNlZE1Oc146rCUtb1bMNPuq6k/view?usp=sharing"

    parquet_path = "data/df_final.parquet"
    w2v_path = "data/GoogleNews-vectors-reduced.bin"

    # Download parquet
    if not os.path.exists(parquet_path):
        gdown.download(parquet_url, parquet_path, quiet=False)

    # Download Word2Vec
    if not os.path.exists(w2v_path):
        gdown.download(word2vec_url, w2v_path, quiet=False)

    # --- Load ---
    df = pd.read_parquet(parquet_path)
    w2v_model = KeyedVectors.load(w2v_path)


    # --- Load Word2Vec Chunks ---
    chunk_ids = [
        "19cn1jpODxFgv1VT6enXI-gF3t4pDTteE",  # word2vec_chunk_hybrid_01.npz 
        "17Vhxw4ErpRDE_rDZYQHNT8BqU0VuPeRM",  # word2vec_chunk_hybrid_02.npz 
        "14pzNVmmuMTOfUy19gPeGWv-dsOy-Ygbn",  # word2vec_chunk_hybrid_03.npz 
        "1VYoGMF5Qv2E9Bn20mRKM-pV61t1j58Pi",  # word2vec_chunk_hybrid_04.npz 
	"1Qnr9IHH8LvnpTtBU2xsfmsTi5KE7nxus",  # word2vec_chunk_hybrid_05.npz 
        "1AFb7dX8Dl1qWyYcYXnkq38PQJtx8nR5E",  # word2vec_chunk_hybrid_06.npz 
        "1z7CJAqkEToG9uOXQVIZ7Kl7MdVFVSlto",  # word2vec_chunk_hybrid_07.npz 
        "1rYVsRFb8lNeS0bpxTOuTDwOyYD1jSEeg",  # word2vec_chunk_hybrid_08.npz 
	"135l1x0JTbOIM3f84CWHRh99pNj0S5VC_",  # word2vec_chunk_hybrid_09.npz 
        "14RxEsGkkfuHxWyQ-K5ufwMzmTHHyBBdL",  # word2vec_chunk_hybrid_10.npz 
        "1o0Mhjms9TT2t1N5AOF4FIyG0XNTn-GKY",  # word2vec_chunk_hybrid_11.npz 
        "18rC3qMOVpW0lSf0jvhRng0cS40zRlBAi",  # word2vec_chunk_hybrid_12.npz 
	"1SCBlEqcQmlnQ4Xh0PFfKchhj4wQHo0gD",  # word2vec_chunk_hybrid_13.npz 
        "1WdKXBsq8hEX7i2HxueYMK1IOqgXYI2TB",  # word2vec_chunk_hybrid_14.npz 
        "1YhBWBm5gVmOTKncrsX4mWEC4gDYE21q-",  # word2vec_chunk_hybrid_15.npz 
        "1NhIxabx9Y73JgPix9XWA5EYLOAoPsSy4",  # word2vec_chunk_hybrid_16.npz 
	"1Vb-KaCrRi5nODkadlQ_4HyMdkQ_zGQST",  # word2vec_chunk_hybrid_17.npz 
        "1p78fcR7977lV32Hsnhq4bHmV-g1GGfvu",  # word2vec_chunk_hybrid_18.npz 
        "1GkcP3HF5idKmKuR9qJoRQ6RXx9V-khUE",  # word2vec_chunk_hybrid_19.npz 
        "1kDzJ4DLkJhpWUFf0DT4lcmwLSDG6GNlW",  # word2vec_chunk_hybrid_20.npz 
	"1VZMw2_v4mLEZsG30Ni8Kfqc9dfNcaYFN",  # word2vec_chunk_hybrid_21.npz 
        "1P_mACznvdZA4d8iwxSxSPksTwZLlRr13",  # word2vec_chunk_hybrid_22.npz 
        "1FGlP7qW2KbOEAAf8ywooIyrNx4dPEMlj",  # word2vec_chunk_hybrid_23.npz 
        "1XGvFFusoGFX1hi9XqmlQkCwHTMV0qbI3",  # word2vec_chunk_hybrid_24.npz 
	"12LDvIZwZic_BtyeqDdNcaeuf863VSAvm",  # word2vec_chunk_hybrid_25.npz 
        "1ivOI8mUAvHwL5ryDkxxYdG_nsSmcleW_",  # word2vec_chunk_hybrid_26.npz 
        "1poJngaPThtWy__NauBZNiUQ5TMLaMNWr"  # word2vec_chunk_hybrid_27.npz 
   
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
