import os
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from glob import glob

DATA_DIR = "data"

def load_model_and_data():
    df_chunks, vec_chunks = [], []

    # Load semua file parquet dan npz dari folder
    parquet_files = sorted(glob(os.path.join(DATA_DIR, "df_final_part_*.parquet")))
    npz_files = sorted(glob(os.path.join(DATA_DIR, "word2vec_chunk_hybrid_*.npz")))

    print(f"📄 Found {len(parquet_files)} parquet files, {len(npz_files)} npz vector files.")

    for pfile, nfile in zip(parquet_files, npz_files):
        print(f"🔹 Loading {pfile} and {nfile}")
        df_chunks.append(pd.read_parquet(pfile))
        vec_chunks.append(sparse.load_npz(nfile).toarray())

    df_final = pd.concat(df_chunks, ignore_index=True)
    vec_all = np.vstack(vec_chunks)

    print(f"✅ DataFrame shape: {df_final.shape}, Vectors shape: {vec_all.shape}")

    if df_final.shape[0] != vec_all.shape[0]:
        raise ValueError(f"❌ Mismatch: {df_final.shape[0]} rows vs {vec_all.shape[0]} vectors")

    model_path = os.path.join(DATA_DIR, "GoogleNews-vectors-reduced.bin")
    print(f"📥 Loading Word2Vec model from {model_path}")
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    return df_final, vec_all, model

def get_vector(text, model):
    words = text.lower().split()
    vectors = [model[w] for w in words if w in model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

def get_recommendation(text, df, vectors, model):
    qvec = get_vector(text, model).reshape(1, -1)
    scores = cosine_similarity(qvec, vectors).flatten()

    if len(scores) != len(df):
        raise ValueError(f"❌ Score length {len(scores)} does not match DataFrame rows {len(df)}")

    df = df.copy()
    df['score'] = scores
    return df.sort_values(by="score", ascending=False).head(10).to_dict(orient="records")
