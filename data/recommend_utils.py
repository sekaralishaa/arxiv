import os
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

# ✅ Lokasi data tetap
DATA_DIR = "/root/rekomendasi-arxiv/data/data"

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
    # 👉 Load query vector dan ubah ke bentuk sparse
    qvec = get_vector(text, model).reshape(1, -1)
    qvec_sparse = sparse.csr_matrix(qvec)

    # 👉 Path untuk file hasil gabungan
    df_path = os.path.join(DATA_DIR, "df_combined.parquet")
    vec_path = os.path.join(DATA_DIR, "matrix_combined.npz")

    # ❗ Validasi file
    if not os.path.exists(df_path) or not os.path.exists(vec_path):
        raise FileNotFoundError("❌ df_combined.parquet atau matrix_combined.npz tidak ditemukan di VPS.")

    # 📥 Load semua data hanya sekali
    print("📥 Load df_combined.parquet dan matrix_combined.npz")
    df = pd.read_parquet(df_path)
    matrix = sparse.load_npz(vec_path)  # tetap dalam bentuk sparse!

    # 🔍 Hitung cosine similarity
    print("🔍 Hitung cosine similarity")
    scores = cosine_similarity(qvec_sparse, matrix).flatten()

    # 🚀 Ambil top 10 berdasarkan skor tertinggi
    df = df.copy()
    df["score"] = scores

    # 🔁 Ambil top 10 & ubah semua kolom jadi serializable
    top10 = df.nlargest(10, "score")

    # ✅ Pastikan semua value bisa diubah ke JSON
    top10_serializable = top10.drop(columns=["vector"], errors="ignore").copy()
    result = top10_serializable.to_dict(orient="records")

    # Convert numpy types to native Python
    for item in result:
        for k, v in item.items():
            if isinstance(v, (np.generic, np.ndarray)):
                item[k] = v.item() if hasattr(v, 'item') else str(v)

    return result
