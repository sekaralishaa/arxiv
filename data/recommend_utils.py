

import os
import zipfile
import requests
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import gdown

DATA_DIR = "data"
MODEL_ZIP_URL = "https://github.com/sekaralishaa/arxiv/releases/download/v1.0/GoogleNews-vectors-reduced.zip"
ZIP_PATH = os.path.join(DATA_DIR, "GoogleNews-vectors-reduced.zip")
BIN_PATH = os.path.join(DATA_DIR, "GoogleNews-vectors-reduced.bin")

# ID Google Drive
PARQUET_IDS = [ "1EXY0tGjiru-brh5blqgbKz-tmgjwAb0K", "1SQwLWQQBHUDQ6IGfWt0OF6dZoDieIgQb", "1PFdf9eFoFVNbvuNiAfJfaINW2BQXlwei", "12MIYAMb4GJ3Cn-jTbKGltDUCaMKb0MpU", "1yu51GxzMXe5FEMrnKN38ZVNQZNNa-xXK", "1xTKztLxAirlM_iHMyKd0ESnOil8wWiEK", "1yCK3utIq2LmGDyEJVNDFouywjwGCd7r1", "1BO3gDGe7LySnEUVA34uDy4xpYKGJsfpO", "1gdglc8pLocVutqLOJ9gMpfKIn_eWtMwi", "1avY9N-U5r9huGhGE4gYQ75eModeNaEjs", "1qw0z9IY2QFBVBuohGNEL58cV46yIkMBu", "1wVBX8HSQMEaWVKO8FRa7Uw7DAD6t2hiJ", "1AVV6f5bu56nHdlRDEyoXTVRp27lCV6Wr", "1D4B47OAZ9CfXFTGJWApYB8DV2mx8CzdU", "1OkBf--RrWyO3qDGL5tKjkZmKz2x43gZ3", "1T2TwCui7-eMfOBhzPsywbhB6Bs_KaIh9", "15Sh_mGkNnLwJ9ioTp1NX76rU0aO_lrMd", "1ebuYe2t_GtrOVW2OOSke23c7wUI0oLML", "1GONjRb-FI0IMhyVB_NXLvL77uB5-4ejq", "1nfLzIqy-JLkZs88vPu7LuoM9XTSvQwrN", "1ugfBEyCMgvJbI6qBEaviKE0W4_lLNm4C", "1jMdb6bfpSW3aFNI6kIAbLmreYVhKRSzZ", "11w8OX2fLtvcvvW1vieketEJsZ3LRlkWI", "1oX-MkEwLil8FEGPgfxcDkOnMliWKt3bu", "1H9YdVSZteZeHjwNAWD6elVWim9Gw-yOg", "1d9xMwtFvvKgFM5T3d-z8NN4ZDyPOZVt3", "1oZWPWNroerQ6gBwm1xWuIUuKN_ong2ZR" ]

NPZ_IDS = [ "19cn1jpODxFgv1VT6enXI-gF3t4pDTteE", "17Vhxw4ErpRDE_rDZYQHNT8BqU0VuPeRM", "14pzNVmmuMTOfUy19gPeGWv-dsOy-Ygbn", "1VYoGMF5Qv2E9Bn20mRKM-pV61t1j58Pi", "1Qnr9IHH8LvnpTtBU2xsfmsTi5KE7nxus", "1AFb7dX8Dl1qWyYcYXnkq38PQJtx8nR5E", "1z7CJAqkEToG9uOXQVIZ7Kl7MdVFVSlto", "1rYVsRFb8lNeS0bpxTOuTDwOyYD1jSEeg", "135l1x0JTbOIM3f84CWHRh99pNj0S5VC_", "14RxEsGkkfuHxWyQ-K5ufwMzmTHHyBBdL", "1o0Mhjms9TT2t1N5AOF4FIyG0XNTn-GKY", "18rC3qMOVpW0lSf0jvhRng0cS40zRlBAi", "1SCBlEqcQmlnQ4Xh0PFfKchhj4wQHo0gD", "1WdKXBsq8hEX7i2HxueYMK1IOqgXYI2TB", "1YhBWBm5gVmOTKncrsX4mWEC4gDYE21q-", "1NhIxabx9Y73JgPix9XWA5EYLOAoPsSy4", "1Vb-KaCrRi5nODkadlQ_4HyMdkQ_zGQST", "1p78fcR7977lV32Hsnhq4bHmV-g1GGfvu", "1GkcP3HF5idKmKuR9qJoRQ6RXx9V-khUE", "1kDzJ4DLkJhpWUFf0DT4lcmwLSDG6GNlW", "1VZMw2_v4mLEZsG30Ni8Kfqc9dfNcaYFN", "1P_mACznvdZA4d8iwxSxSPksTwZLlRr13", "1FGlP7qW2KbOEAAf8ywooIyrNx4dPEMlj", "1XGvFFusoGFX1hi9XqmlQkCwHTMV0qbI3", "12LDvIZwZic_BtyeqDdNcaeuf863VSAvm", "1ivOI8mUAvHwL5ryDkxxYdG_nsSmcleW_", "1poJngaPThtWy__NauBZNiUQ5TMLaMNWr" ]

def download_from_gdrive_if_needed(path, file_id):
    if not os.path.exists(path):
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"📥 Downloading {os.path.basename(path)} from Google Drive...")
        gdown.download(url, path, quiet=False, fuzzy=True)

def download_and_extract_model_if_needed():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(BIN_PATH):
        print("📥 Downloading Word2Vec zip from GitHub Releases...")
        r = requests.get(MODEL_ZIP_URL)
        with open(ZIP_PATH, 'wb') as f:
            f.write(r.content)

        print("📦 Extracting Word2Vec zip...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print("✅ Extraction complete.")
    else:
        print("📦 Word2Vec model already exists, skipping download.")

def load_model():
    download_and_extract_model_if_needed()
    print(f"📥 Loading Word2Vec model from {BIN_PATH}")
    return KeyedVectors.load_word2vec_format(BIN_PATH, binary=True)

def get_vector(text, model):
    words = text.lower().split()
    vectors = [model[w] for w in words if w in model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

def get_recommendation(text, model):
    qvec = get_vector(text, model).reshape(1, -1)
    results = []

    for i in range(1, 28):
        parquet_path = os.path.join(DATA_DIR, f"df_final_part_{i:02d}.parquet")
        npz_path = os.path.join(DATA_DIR, f"word2vec_chunk_hybrid_{i:02d}.npz")

        download_from_gdrive_if_needed(parquet_path, PARQUET_IDS[i - 1])
        download_from_gdrive_if_needed(npz_path, NPZ_IDS[i - 1])

        df_chunk = pd.read_parquet(parquet_path)
        vec_chunk = sparse.load_npz(npz_path).toarray()

        scores = cosine_similarity(qvec, vec_chunk).flatten()
        df_chunk = df_chunk.copy()
        df_chunk['score'] = scores
        top = df_chunk.sort_values(by='score', ascending=False).head(10)
        results.append(top)

    df_all = pd.concat(results, ignore_index=True)
    return df_all.sort_values(by="score", ascending=False).head(10).to_dict(orient="records")

