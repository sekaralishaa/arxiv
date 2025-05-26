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
    chunk_ids = [
    "1EXY0tGjiru-brh5blqgbKz-tmgjwAb0K",  # df_final_part_01.parquet
    "1SQwLWQQBHUDQ6IGfWt0OF6dZoDieIgQb",  # df_final_part_02.parquet
    "1PFdf9eFoFVNbvuNiAfJfaINW2BQXlwei",  # df_final_part_03.parquet
    "12MIYAMb4GJ3Cn-jTbKGltDUCaMKb0MpU",  # df_final_part_04.parquet
    "1yu51GxzMXe5FEMrnKN38ZVNQZNNa-xXK",  # df_final_part_05.parquet
    "1xTKztLxAirlM_iHMyKd0ESnOil8wWiEK",  # df_final_part_06.parquet
    "1yCK3utIq2LmGDyEJVNDFouywjwGCd7r1",  # df_final_part_07.parquet
    "1BO3gDGe7LySnEUVA34uDy4xpYKGJsfpO",  # df_final_part_08.parquet
    "1gdglc8pLocVutqLOJ9gMpfKIn_eWtMwi",  # df_final_part_09.parquet
    "1avY9N-U5r9huGhGE4gYQ75eModeNaEjs",  # df_final_part_10.parquet
    "1qw0z9IY2QFBVBuohGNEL58cV46yIkMBu",  # df_final_part_11.parquet
    "1wVBX8HSQMEaWVKO8FRa7Uw7DAD6t2hiJ",  # df_final_part_12.parquet
    "1AVV6f5bu56nHdlRDEyoXTVRp27lCV6Wr",  # df_final_part_13.parquet
    "1D4B47OAZ9CfXFTGJWApYB8DV2mx8CzdU",  # df_final_part_14.parquet
    "1OkBf--RrWyO3qDGL5tKjkZmKz2x43gZ3",  # df_final_part_15.parquet
    "1T2TwCui7-eMfOBhzPsywbhB6Bs_KaIh9",  # df_final_part_16.parquet
    "15Sh_mGkNnLwJ9ioTp1NX76rU0aO_lrMd",  # df_final_part_17.parquet
    "1ebuYe2t_GtrOVW2OOSke23c7wUI0oLML",  # df_final_part_18.parquet
    "1GONjRb-FI0IMhyVB_NXLvL77uB5-4ejq",  # df_final_part_19.parquet
    "1nfLzIqy-JLkZs88vPu7LuoM9XTSvQwrN",  # df_final_part_20.parquet
    "1ugfBEyCMgvJbI6qBEaviKE0W4_lLNm4C",  # df_final_part_21.parquet
    "1jMdb6bfpSW3aFNI6kIAbLmreYVhKRSzZ",  # df_final_part_22.parquet
    "11w8OX2fLtvcvvW1vieketEJsZ3LRlkWI",  # df_final_part_23.parquet
    "1oX-MkEwLil8FEGPgfxcDkOnMliWKt3bu",  # df_final_part_24.parquet
    "1H9YdVSZteZeHjwNAWD6elVWim9Gw-yOg",  # df_final_part_25.parquet
    "1d9xMwtFvvKgFM5T3d-z8NN4ZDyPOZVt3",  # df_final_part_26.parquet
    "1oZWPWNroerQ6gBwm1xWuIUuKN_ong2ZR"   # df_final_part_27.parquet
    
    ]

    chunks = []
    for i, chunk_id in enumerate(chunk_ids, start=1):
    chunk_path = f"data/df_final_part_{i:02d}.parquet"
    chunk_url = f"https://drive.google.com/uc?id={chunk_id}"

        if not os.path.exists(chunk_path):
            gdown.download(chunk_url, chunk_path, quiet=False)

    chunks.append(pd.read_parquet(chunk_path))

    df = pd.concat(chunks, ignore_index=True)
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
