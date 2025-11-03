import os
import time
import logging
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import joblib
import ast
import numpy as np

# ---------------- Logging Setup ---------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# ---------------- Load .env ---------------- #
load_dotenv()

# ---------------- Config ---------------- #
CONFIG = {
    "data_dir": "../data/ml-32m",
    "embeddings_dir": "../data/embeddings",
    "movies_file": "movies.csv",
    "tags_file": "tags.csv",
    "embeddings_file": "movie_embeddings.csv",
    "embedding_model": "text-embedding-3-small"
}

# ---------------- Core Functions ---------------- #
def load_data(movies_path: str, tags_path: str) -> pd.DataFrame:
    """Load and merge movies and tags CSVs into a DataFrame with text features."""
    if not (os.path.exists(movies_path) and os.path.exists(tags_path)):
        raise FileNotFoundError("Movies or Tags CSV files not found.")

    logging.info("📂 Loading movies and tags data...")
    movies = pd.read_csv(movies_path)
    tags = pd.read_csv(tags_path)

    # Validate required columns
    required_movies_cols = ['movieId', 'title', 'genres']
    required_tags_cols = ['movieId', 'tag']
    missing_movies = [c for c in required_movies_cols if c not in movies.columns]
    missing_tags = [c for c in required_tags_cols if c not in tags.columns]
    if missing_movies:
        raise ValueError(f"Missing required columns in movies CSV: {missing_movies}")
    if missing_tags:
        raise ValueError(f"Missing required columns in tags CSV: {missing_tags}")

    # Aggregate tags
    tags_agg = (
        tags.groupby('movieId')['tag']
        .agg(lambda x: '|'.join(str(tag) for tag in set(x) if pd.notnull(tag)))
        .reset_index()
    )

    # Merge
    movies_tags = movies.merge(tags_agg, on='movieId', how='left')
    movies_tags['text_features'] = movies_tags['tag'].fillna('') + '|' + movies_tags['genres'].fillna('')

    logging.info(f"✅ Data merged: {movies_tags.shape[0]} rows")
    return movies_tags


def generate_embeddings(df: pd.DataFrame, model: str = None) -> pd.DataFrame:
    """Generate OpenAI embeddings for the 'text_features' column."""
    model = model or CONFIG['embedding_model']
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")

    client = OpenAI(api_key=api_key)

    def get_embedding(text):
        try:
            return client.embeddings.create(model=model, input=text).data[0].embedding
        except Exception as e:
            logging.error(f"❌ Failed to generate embedding for text: {text[:30]}... | {e}")
            return None

    logging.info(f"🧠 Generating embeddings using model '{model}' (this may take a while)...")
    start_time = time.time()
    df['embedding'] = df['text_features'].apply(get_embedding)
    elapsed = time.time() - start_time
    logging.info(f"✅ Embeddings generated in {elapsed:.2f}s")

    return df[['movieId', 'title', 'embedding', 'text_features']]


def create_embeddings(
    movies_path: str | None = None,
    tags_path: str | None = None,
    embeddings_path: str | None = None,
    model: str | None = None,
    retrain: bool = False
) -> pd.DataFrame:
    """Load or create embeddings for content-based recommendation."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    movies_path = movies_path or os.path.join(base_dir, CONFIG["data_dir"], CONFIG["movies_file"])
    tags_path = tags_path or os.path.join(base_dir, CONFIG["data_dir"], CONFIG["tags_file"])
    embeddings_path = embeddings_path or os.path.join(base_dir, CONFIG["embeddings_dir"], CONFIG["embeddings_file"])
    model = model or CONFIG["embedding_model"]

    if not retrain and os.path.exists(embeddings_path):
        try:
            logging.info(f"📦 Loading existing embeddings from {embeddings_path}")
            df = pd.read_csv(embeddings_path)
            if 'embedding' in df.columns:
                logging.info(f"✅ Embeddings loaded successfully ({df.shape[0]} rows)")
                return df
        except Exception as e:
            logging.warning(f"⚠️  Failed to load embeddings ({e}). Recreating...")

    logging.info("🚀 Creating new embeddings...")
    df = load_data(movies_path, tags_path)
    embedding_df = generate_embeddings(df, model=model)
    os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
    embedding_df.to_csv(embeddings_path, index=False)
    logging.info(f"💾 Embeddings saved → {embeddings_path}")

    return embedding_df

def embedding_matrix(df: pd.DataFrame,
                     matrix_path: str | None = None,
                     rebuild: bool = False) -> np.ndarray:
    """Create or load existing embeddings into a numeric NumPy matrix."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    matrix_path = matrix_path or os.path.join(base_dir, CONFIG["embeddings_dir"], "embedding_matrix.npy")

    if not rebuild and os.path.exists(matrix_path):
        try:
            logging.info(f"📦 Loading existing embedding matrix from {matrix_path}")
            emb_matrix = np.load(matrix_path)
            logging.info(f"✅ Embedding matrix loaded successfully (shape: {emb_matrix.shape})")
            return emb_matrix
        except Exception as e:
            logging.warning(f"⚠️ Failed to load embedding matrix ({e}). Rebuilding...")

    if 'embedding' not in df.columns:
        raise ValueError("DataFrame must contain an 'embedding' column.")

    if isinstance(df['embedding'].iloc[0], str):
        df['embedding'] = df['embedding'].apply(ast.literal_eval)

    emb_matrix = np.array(df['embedding'].to_list(), dtype=float)
    os.makedirs(os.path.dirname(matrix_path), exist_ok=True)
    np.save(matrix_path, emb_matrix)
    logging.info(f"💾 Embedding matrix saved → {matrix_path}")

    return emb_matrix


# ---------------- CLI Entry Point ---------------- #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create or load movie embeddings for content-based recommendation.")
    parser.add_argument("--retrain", action="store_true", help="Force regenerating embeddings even if they exist.")
    parser.add_argument("--model", type=str, help="OpenAI embedding model to use.")
    args = parser.parse_args()

    create_embeddings(retrain=args.retrain, model=args.model)
