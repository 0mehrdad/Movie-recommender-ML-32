import numpy as np
import pandas as pd
import logging
from typing import Optional, List


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

    return pd.DataFrame(preds[:top_n], columns=["movieId", "predicted_rating"])


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
