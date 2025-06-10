import os
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

DATA_DIR = "/root/rekomendasi-arxiv/data/data"  # lokasi fix di VPS kamu

def load_model():
    kv_path = os.path.join(DATA_DIR, "GoogleNews-vectors-reduced.kv")
    npy_path = kv_path + ".vectors.npy"
    
    if not os.path.exists(kv_path) or not os.path.exists(npy_path):
        raise FileNotFoundError("❌ Model Word2Vec belum tersedia di VPS. Pastikan file .kv dan .kv.vectors.npy ada.")

    print(f"✅ Loading Word2Vec model from {kv_path}")
    return KeyedVectors.load(kv_path)

def get_vector(text, model):
    words = text.lower().split()
    vectors = [model[w] for w in words if w in model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

def get_recommendation(text, model):
    qvec_dense = get_vector(text, model).reshape(1, -1)
    qvec_sparse = sparse.csr_matrix(qvec_dense)  # Ubah query ke sparse
    top_results = []

    for i in range(1, 28):
        parquet_path = os.path.join(DATA_DIR, "df_final.parquet")
        npz_path = os.path.join(DATA_DIR, f"word2vec_chunk_hybrid_{i:02d}.npz")

        if not os.path.exists(parquet_path) or not os.path.exists(npz_path):
            print(f"⚠ Melewati chunk {i:02d}, file tidak ditemukan.")
            continue

        df_chunk = pd.read_parquet(parquet_path)
        vec_chunk = sparse.load_npz(npz_path)  # tetap dalam bentuk sparse

        scores = cosine_similarity(qvec_sparse, vec_chunk).flatten()

        df_chunk = df_chunk.copy()
        df_chunk["score"] = scores
        top_chunk = df_chunk.nlargest(10, "score")
        top_results.append(top_chunk)

    if not top_results:
        return []

    df_all = pd.concat(top_results, ignore_index=True)
    return df_all.nlargest(10, "score").to_dict(orient="records")
