import numpy as np
import pandas as pd
import logging
from typing import Optional, List
import requests
import logging
import re
# -------------------------------------------------------------
# Recommendation Utilities
# -------------------------------------------------------------

def recommend_svd_ids(
    df: pd.DataFrame,
    model,
    user_id: int,
    candidate_movie_ids: Optional[List[int]] = None,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Recommend top-N movies for a user using an SVD model.
    If candidate_movie_ids is None, predict for all unseen movies.
    """
    if user_id not in df["userId"].unique():
        raise ValueError(f"User {user_id} not found in dataset.")

    watched = set(df.loc[df["userId"] == user_id, "movieId"])
    pool = (
        [m for m in df["movieId"].unique() if m not in watched]
        if candidate_movie_ids is None
        else [m for m in candidate_movie_ids if m not in watched]
    )

    preds = [(m, round(model.predict(user_id, m).est, 3)) for m in pool]
    preds.sort(key=lambda x: x[1], reverse=True)
    df_pred = pd.DataFrame(preds[:top_n], columns=["movieId", "predicted_rating"])
    df_pred = df_pred.merge(df[["movieId", "title"]].drop_duplicates(), on="movieId", how="left")
    return df_pred



# -------------------------------------------------------------
# Embedding Utilities
# -------------------------------------------------------------

def _cosine_sim_rowwise(emb_matrix: np.ndarray, idx: int) -> np.ndarray:
    """Compute cosine similarity between one row vector and all others."""
    target = np.nan_to_num(emb_matrix[idx])
    emb = np.nan_to_num(emb_matrix)
    denom = np.linalg.norm(emb, axis=1) * (np.linalg.norm(target) + 1e-12)
    sims = np.dot(emb, target) / np.clip(denom, 1e-12, None)
    return np.nan_to_num(sims)


def similar_movies(
    movie_id: int,
    emb_df: pd.DataFrame,
    emb_matrix: np.ndarray,
    top_k: int = 50
) -> pd.DataFrame:
    """
    Return top-K most similar movies based on cosine similarity.
    Requires emb_df to contain columns ['movieId', 'title'] aligned with emb_matrix rows.
    """
    matches = emb_df.index[emb_df["movieId"] == movie_id]
    if len(matches) == 0:
        raise ValueError(f"movieId {movie_id} not found in embeddings.")

    idx = int(matches[0])
    sims = _cosine_sim_rowwise(emb_matrix, idx)
    order = np.argsort(sims)[::-1]
    order = order[order != idx]
    top_idx = order[:top_k]

    out = emb_df.iloc[top_idx][["movieId", "title"]].copy()
    out["similarity"] = sims[top_idx]
    return out.reset_index(drop=True)

def get_poster(movieId: int, links_df: pd.DataFrame, api_key: str) -> str | None:
    """
    Fetch the poster URL for a movieId using TMDB API.
    Returns None if not found or on error.
    """
    tmdbId = get_tmdb_id(links_df, movieId)
    try:
        resp = requests.get(
            f"https://api.themoviedb.org/3/search/movie/{tmdbId}",
            params={"api_key": api_key},
            timeout=5
        )
        resp.raise_for_status()
        data = resp.json().get("results", [])
        if not data:
            return None

        poster_path = data[0].get("poster_path")
        if not poster_path:
            return None

        return f"https://image.tmdb.org/t/p/w500{poster_path}"

    except requests.RequestException as e:
        logging.warning(f"TMDB fetch failed for '{movieId}': {e}")
        return None
    
def get_tmdb_id(links_df: pd.DataFrame, movie_id: int) -> int | None:
    """
    Return the TMDB ID for a given MovieLens movieId.
    Returns None if not found or invalid.

    Args:
        links_df: DataFrame containing 'movieId' and 'tmdbId' columns.
        movie_id: The MovieLens movieId.

    Returns:
        int | None: TMDB ID if available, else None.
    """
    try:
        row = links_df.loc[links_df["movieId"] == movie_id, "tmdbId"]
        if row.empty:
            logging.debug(f"movieId {movie_id} not found in links_df.")
            return None

        tmdb_id = row.iloc[0]
        if pd.isna(tmdb_id):
            logging.debug(f"movieId {movie_id} has no tmdbId.")
            return None

        return int(tmdb_id)
    except Exception as e:
        logging.warning(f"Failed to get TMDB ID for movieId {movie_id}: {e}")
        return None

