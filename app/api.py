from fastapi import FastAPI, HTTPException
from app.train_svd import setup_logger, load_config, load_data, train_svd
from app.create_embeddings import load_data as load_movie_tags, generate_embeddings, embedding_matrix
from app.utils import recommend_svd_ids, similar_movies, get_poster
import pandas as pd
import numpy as np
import os
import joblib
import logging


app = FastAPI(title="Movie Recommender API")

# ---- Startup: load all assets once ----
setup_logger("logs/api.log")
cfg = load_config()
base = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(base, cfg["model_dir"], cfg["model_file"])
ratings_path = os.path.join(base, cfg["data_dir"], cfg["ratings_file"])
movies_path = os.path.join(base, cfg["data_dir"], cfg["movies_file"])
tags_path = os.path.join(base, cfg["data_dir"], cfg["tags_file"])
embeddings_dir = os.path.join(base, cfg["embeddings_dir"])
emb_csv = os.path.join(embeddings_dir, cfg["embeddings_file"])
emb_mat_path = os.path.join(embeddings_dir, "embedding_matrix.npy")
links_path = os.path.join(base, cfg["data_dir"], "links.csv")


# --- Load or train SVD ---
if os.path.exists(model_path):
    model = joblib.load(model_path)
    logging.info("Loaded existing SVD model.")
else:
    df_train = load_data(movies_path, ratings_path)
    model = train_svd(df_train, cfg["svd_params"], model_path, cfg["test_size"], cfg["random_state"])
    logging.info("Trained new SVD model.")

# --- Load or create embeddings ---
if os.path.exists(emb_csv) and os.path.exists(emb_mat_path):
    emb_df = pd.read_csv(emb_csv)
    emb_matrix_np = np.load(emb_mat_path)
    logging.info("Loaded existing embeddings.")
else:
    df_emb = load_movie_tags(movies_path, tags_path)
    df_emb = generate_embeddings(df_emb, cfg["embedding_model"], None, emb_csv)
    emb_matrix_np = embedding_matrix(df_emb, emb_mat_path)
    emb_df = pd.read_csv(emb_csv)
    logging.info("Generated new embeddings.")

df_all = load_data(movies_path, ratings_path)
if os.path.exists(links_path):
    links_df = pd.read_csv(links_path)
else:
    logging.warning("links.csv not found — TMDB mapping disabled.")
    links_df = pd.DataFrame()

# ---- Endpoints ----

@app.get("/")
def root():
    return {"message": "Movie Recommendation API is live."}


@app.get("/recommend/user/{user_id}")
def recommend_for_user(user_id: int, top_n: int = 10):
    """Return top-N SVD recommendations for a user."""
    try:
        recs = recommend_svd_ids(df_all, model, user_id, top_n=top_n)
        recs ['poster'] = recs['movieId'].apply(lambda x: get_poster(x, links_df, os.getenv("TMDB_API_KEY")))
        return recs.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/similar/{movie_id}")
def similar(movie_id: int, top_k: int = 10):
    """Return top-K similar movies based on embeddings."""
    try:
        sims = similar_movies(movie_id, emb_df, emb_matrix_np, top_k=top_k)
        sims['poster'] = sims['movieId'].apply(lambda x: get_poster(x, links_df, os.getenv("TMDB_API_KEY")))
        return sims.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
