import os
import argparse
import logging
import joblib
import pandas as pd
import time
import yaml
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split


def setup_logger(log_path="logs/train_svd.log"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, mode="a")]
    )


def load_config(cfg_path="config.yaml"):
    if not os.path.exists(cfg_path):
        raise FileNotFoundError("Missing config.yaml file.")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def load_data(movies_path: str, ratings_path: str) -> pd.DataFrame:
    if not (os.path.exists(ratings_path) and os.path.exists(movies_path)):
        raise FileNotFoundError("Ratings or movies CSV files not found.")
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)

    required_movies_cols = {"movieId", "title"}
    required_ratings_cols = {"movieId", "userId", "rating"}
    if not required_movies_cols.issubset(movies.columns):
        raise ValueError("Movies CSV missing required columns.")
    if not required_ratings_cols.issubset(ratings.columns):
        raise ValueError("Ratings CSV missing required columns.")

    df = ratings.merge(movies, on="movieId")
    logging.info(f"Dataset merged: {df.shape[0]} rows")
    return df


def train_svd(df: pd.DataFrame, params: dict, model_path: str, test_size: float, random_state: int):
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(df[["userId", "movieId", "rating"]], reader)
    trainset, testset = train_test_split(data, test_size=test_size, random_state=random_state)

    model = SVD(**params, random_state=random_state)
    start = time.time()
    model.fit(trainset)
    elapsed = time.time() - start
    logging.info(f"Training done in {elapsed:.2f}s")

    preds = model.test(testset)
    rmse = accuracy.rmse(preds, verbose=False)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    logging.info(f"Model saved -> {model_path} | RMSE: {rmse:.4f}")
    return model

'''
def main():
    setup_logger()
    cfg = load_config()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, cfg["model_dir"], cfg["model_file"])
    ratings_path = os.path.join(base_dir, cfg["data_dir"], cfg["ratings_file"])
    movies_path = os.path.join(base_dir, cfg["data_dir"], cfg["movies_file"])

    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--seed", type=int, default=cfg["random_state"])
    args = parser.parse_args()

    if not args.retrain and os.path.exists(model_path):
        logging.info("Loading cached SVD model...")
        model = joblib.load(model_path)
        logging.info("Model loaded successfully.")
        return model
    df = load_data(movies_path, ratings_path)
    return train_svd(df, cfg["svd_params"], model_path, cfg["test_size"], args.seed)
'''
