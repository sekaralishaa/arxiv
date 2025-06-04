import os
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import heapq
import gdown
import requests
from tqdm import tqdm

DATA_DIR = "data"
GITHUB_BASE = "https://github.com/sekaralishaa/arxiv/releases/download/v1.3"

NPZ_IDS = [
    "19cn1jpODxFgv1VT6enXI-gF3t4pDTteE", "17Vhxw4ErpRDE_rDZYQHNT8BqU0VuPeRM",
    "14pzNVmmuMTOfUy19gPeGWv-dsOy-Ygbn", "1VYoGMF5Qv2E9Bn20mRKM-pV61t1j58Pi",
    "1Qnr9IHH8LvnpTtBU2xsfmsTi5KE7nxus", "1AFb7dX8Dl1qWyYcYXnkq38PQJtx8nR5E",
    "1z7CJAqkEToG9uOXQVIZ7Kl7MdVFVSlto", "1rYVsRFb8lNeS0bpxTOuTDwOyYD1jSEeg",
    "135l1x0JTbOIM3f84CWHRh99pNj0S5VC_", "14RxEsGkkfuHxWyQ-K5ufwMzmTHHyBBdL",
    "1o0Mhjms9TT2t1N5AOF4FIyG0XNTn-GKY", "18rC3qMOVpW0lSf0jvhRng0cS40zRlBAi",
    "1SCBlEqcQmlnQ4Xh0PFfKchhj4wQHo0gD", "1WdKXBsq8hEX7i2HxueYMK1IOqgXYI2TB",
    "1YhBWBm5gVmOTKncrsX4mWEC4gDYE21q-", "1NhIxabx9Y73JgPix9XWA5EYLOAoPsSy4",
    "1Vb-KaCrRi5nODkadlQ_4HyMdkQ_zGQST", "1p78fcR7977lV32Hsnhq4bHmV-g1GGfvu",
    "1GkcP3HF5idKmKuR9qJoRQ6RXx9V-khUE", "1kDzJ4DLkJhpWUFf0DT4lcmwLSDG6GNlW",
    "1VZMw2_v4mLEZsG30Ni8Kfqc9dfNcaYFN", "1P_mACznvdZA4d8iwxSxSPksTwZLlRr13",
    "1FGlP7qW2KbOEAAf8ywooIyrNx4dPEMlj", "1XGvFFusoGFX1hi9XqmlQkCwHTMV0qbI3",
    "12LDvIZwZic_BtyeqDdNcaeuf863VSAvm", "1ivOI8mUAvHwL5ryDkxxYdG_nsSmcleW_",
    "1poJngaPThtWy__NauBZNiUQ5TMLaMNWr"
]

MODEL_ID = "1Mzvz1nApC8T5-YRmHoXU1OkdS29woyPm"
MODEL_NPY_ID = "1Wq_J3AD8HLirsJE8ew0ehlMCLMTfMjXZ"


def download_if_not_exists(path, source):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) >= 1000:
        return

    if source.startswith("http"):
        response = requests.get(source, stream=True)
        with open(path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    else:
        gdown.download(f"https://drive.google.com/uc?id={source}", path, quiet=False)


def load_model():
    kv_path = os.path.join(DATA_DIR, "GoogleNews-vectors-reduced.kv")
    npy_path = kv_path + ".vectors.npy"
    download_if_not_exists(kv_path, MODEL_ID)
    download_if_not_exists(npy_path, MODEL_NPY_ID)
    return KeyedVectors.load(kv_path)


def get_vector(text, model):
    words = text.lower().split()
    vectors = [model[w] for w in words if w in model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)


def get_recommendation(text, model):
    qvec = get_vector(text, model).reshape(1, -1)
    top_k = []  # min-heap (score, row)

    for i in range(1, 28):
        parquet_filename = f"df_final_part_{i:02d}.parquet"
        parquet_path = os.path.join(DATA_DIR, parquet_filename)
        parquet_url = f"{GITHUB_BASE}/{parquet_filename}"
        download_if_not_exists(parquet_path, parquet_url)

        npz_filename = f"word2vec_chunk_hybrid_{i:02d}.npz"
        npz_path = os.path.join(DATA_DIR, npz_filename)
        download_if_not_exists(npz_path, NPZ_IDS[i - 1])

        df_chunk = pd.read_parquet(parquet_path)
        vec_chunk = sparse.load_npz(npz_path)

        for idx in range(vec_chunk.shape[0]):
            score = cosine_similarity(qvec, vec_chunk[idx]).flatten()[0]
            row = df_chunk.iloc[idx].to_dict()
            row["score"] = score

            if len(top_k) < 10:
                heapq.heappush(top_k, (score, row))
            else:
                heapq.heappushpop(top_k, (score, row))

    return sorted([item for _, item in top_k], key=lambda x: x["score"], reverse=True)
