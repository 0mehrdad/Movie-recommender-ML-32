import os
import argparse
import logging
import joblib
import pandas as pd
import time
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

# ---------------- Logging Setup ---------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# ---------------- Config ---------------- #
CONFIG = {
    "data_dir": "../data/ml-32m",
    "model_dir": "../models",
    "ratings_file": "ratings.csv",
    "movies_file": "movies.csv",
    "model_file": "svd_model.pkl",
    "test_size": 0.2,
    "random_state": 42,
    "svd_params": {
        "n_factors": 100,
        "reg_all": 0.02
    }
}


# ---------------- Core Function ---------------- #
def load_data(movies_path: str, ratings_path: str) -> pd.DataFrame:
    """Load and merge movies and ratings CSVs into a DataFrame."""
    if not (os.path.exists(ratings_path) and os.path.exists(movies_path)):
            raise FileNotFoundError("Ratings or movies CSV files not found.")
    logging.info("📂 Loading and merging datasets...")
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)

    # Validate columns
    required_movies_cols = {"movieId", "title"}
    required_ratings_cols = {"movieId", "userId", "rating"}
    if not required_movies_cols.issubset(movies.columns):
        raise ValueError("Movies CSV missing required columns.")
    if not required_ratings_cols.issubset(ratings.columns):
        raise ValueError("Ratings CSV missing required columns.")

    df = ratings.merge(movies, on="movieId")
    logging.info(f"✅ Dataset merged: {df.shape[0]} rows")
    return df

def svd_model(
    df: pd.DataFrame | None = None,
    ratings_path: str | None = None,
    movies_path: str | None = None,
    model_path: str | None = None,
    retrain: bool = False,
    random_state: int | None = None
) -> SVD:
    """
    Train or load an SVD recommendation model.

    Args:
        df: Optional DataFrame with ['userId', 'title', 'rating'].
        ratings_path: Optional path to ratings CSV.
        movies_path: Optional path to movies CSV.
        model_path: Optional path to save/load model.
        retrain: If True, forces retraining.
        random_state: Optional random seed for reproducibility.

    Returns:
        Trained or loaded SVD model.
    """

    # Resolve paths dynamically
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ratings_path = ratings_path or os.path.join(base_dir, CONFIG["data_dir"], CONFIG["ratings_file"])
    movies_path = movies_path or os.path.join(base_dir, CONFIG["data_dir"], CONFIG["movies_file"])
    model_path = model_path or os.path.join(base_dir, CONFIG["model_dir"], CONFIG["model_file"])
    random_state = random_state or CONFIG["random_state"]

    # Load existing model if available
    if not retrain and os.path.exists(model_path):
        logging.info("📦 Loading existing SVD model — this might take a few seconds...")
        try:
            start_time = time.time()
            model = joblib.load(model_path)
            load_time = time.time() - start_time
            logging.info(f"✅ Model loaded from {model_path} in {load_time:.2f}s")
            return model
        except Exception as e:
            logging.warning(f"⚠️  Failed to load model ({e}). Retraining...")

    # Load datasets if df not provided
    if df is None:
        df = load_data(movies_path, ratings_path)



    # Prepare and train SVD
    logging.info("⚙️ Preparing dataset for SVD training (this may take a couple of minutes)...")
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(df[["userId", "title", "rating"]], reader)
    trainset, testset = train_test_split(data, test_size=CONFIG["test_size"], random_state=random_state)

    logging.info("🧠 Initializing SVD model with parameters: "
                 f"{CONFIG['svd_params']} | random_state={random_state}")
    model = SVD(**CONFIG["svd_params"], random_state=random_state)

    logging.info("🚀 Starting SVD training — please wait...")
    start_time = time.time()
    model.fit(trainset)
    training_time = time.time() - start_time
    logging.info(f"✅ Training completed in {training_time:.2f}s — evaluating model...")

    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    logging.info(f"💾 Model saved → {model_path} | RMSE: {rmse:.4f}")

    return model


# ---------------- CLI Entry Point ---------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or load an SVD recommender model.")
    parser.add_argument("--retrain", action="store_true", help="Force retraining even if model exists.")
    parser.add_argument("--seed", type=int, help="Override random seed for reproducibility.")
    args = parser.parse_args()

    svd_model(retrain=args.retrain, random_state=args.seed)
