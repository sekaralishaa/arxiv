import os
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

# Lokasi file hasil penggabungan
DATA_DIR = "/root/rekomendasi-arxiv/data/data"

# ===============================
# Load Word2Vec Model (format .kv)
# ===============================
def load_model():
    model_path = os.path.join(DATA_DIR, "GoogleNews-vectors-negative3001.bin")
    print("📥 Loading Word2Vec model (.bin)")
    return KeyedVectors.load_word2vec_format(model_path, binary=True)

# ===============================
# Clean Text (untuk konsistensi)
# ===============================
def clean_text(text):
    import re
    import string
    if pd.isna(text):
        return ""
    text = re.sub(r"\\$\([^)]*\)\\$", "", text)
    text = text.replace("\n", " ").replace("-", " ")
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\\s+", " ", text).strip().lower()
    return text

# ===============================
# Get Word2Vec Vector dari input user
# ===============================
def get_vector(text, model):
    words = text.lower().split()
    vectors = [model[word] for word in words if word in model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size, dtype=np.float32)

# ===============================
# Main Recommendation Function
# ===============================
def recommend_articles(user_title, user_keywords, user_category, model, chunk_size=100000):
    # Gabungkan & bersihkan input
    combined_input = clean_text(f"{user_title} {user_keywords.replace(',', ' ')} {user_category}")
    query_vec = get_vector(combined_input, model).reshape(1, -1)

    df_path = os.path.join(DATA_DIR, "df_combined.parquet")
    vec_path = os.path.join(DATA_DIR, "matrix_combined.npz")

    if not os.path.exists(df_path) or not os.path.exists(vec_path):
        raise FileNotFoundError("❌ df_combined.parquet atau matrix_combined.npz tidak ditemukan.")

    print("📥 Load df_combined & matrix_combined")
    df = pd.read_parquet(df_path)
    matrix = sparse.load_npz(vec_path)

    print("🔍 Hitung cosine similarity per chunk")
    all_scores = []
    all_indexes = []

    start_idx = 0
    while start_idx < matrix.shape[0]:
        end_idx = min(start_idx + chunk_size, matrix.shape[0])
        chunk = matrix[start_idx:end_idx]
        sim = cosine_similarity(query_vec, chunk).flatten()
        all_scores.extend(sim)
        all_indexes.extend(range(start_idx, end_idx))
        start_idx = end_idx

    print("✅ Gabungkan dan sortir hasil")
    df_chunked = df.iloc[all_indexes].copy()
    df_chunked["similarity_score"] = all_scores
    top10 = df_chunked.nlargest(10, "similarity_score")

    # Konversi nilai agar JSON-serializable
    def safe_convert(v):
        if isinstance(v, (np.generic, np.bool_)):
            return v.item()
        elif isinstance(v, np.ndarray):
            return v.tolist()
        return v

    result = top10[["title", "authors", "categories_clean", "abstract", "doi", "similarity_score"]].to_dict(orient="records")

    for item in result:
        for k in item:
            item[k] = safe_convert(item[k])

    return result

