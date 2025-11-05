import os
import time
import logging
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import joblib
import numpy as np
import yaml
import ast

def setup_logger(log_path="logs/embed.log"):
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

def load_data(movies_path, tags_path):
    movies = pd.read_csv(movies_path)
    tags = pd.read_csv(tags_path)

    if not {'movieId', 'title', 'genres'}.issubset(movies.columns):
        raise ValueError("Movies CSV missing required columns.")
    if not {'movieId', 'tag'}.issubset(tags.columns):
        raise ValueError("Tags CSV missing required columns.")

    tags_agg = tags.groupby('movieId')['tag'].agg(lambda x: '|'.join(set(x.dropna()))).reset_index()
    df = movies.merge(tags_agg, on='movieId', how='left')
    df['text_features'] = (df['tag'].fillna('') + '|' + df['genres'].fillna('')).str.strip('|')
    return df[['movieId', 'title', 'text_features']]

def batch(iterable, n=100):
    l = len(iterable)
    for i in range(0, l, n):
        yield iterable[i:min(i + n, l)]

def generate_embeddings(df, model, client, cache_path):
    if os.path.exists(cache_path):
        cached = pd.read_csv(cache_path)
        cached_ids = set(cached['movieId'])
        df = df[~df['movieId'].isin(cached_ids)]
        all_data = pd.concat([cached, df], ignore_index=True)
    else:
        all_data = df.copy()

    new_rows = []
    for chunk in batch(df.to_dict('records'), 100):
        inputs = [r['text_features'] for r in chunk]
        try:
            response = client.embeddings.create(model=model, input=inputs)
            for record, emb in zip(chunk, response.data):
                record['embedding'] = emb.embedding
                new_rows.append(record)
        except Exception as e:
            logging.error(f"Embedding batch failed: {e}")

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        result = pd.concat([all_data, new_df], ignore_index=True).drop_duplicates('movieId')
        result.to_csv(cache_path, index=False)
        return result
    else:
        return all_data

def embedding_matrix(df, matrix_path):
    if 'embedding' not in df.columns:
        raise ValueError("Missing 'embedding' column.")
    if isinstance(df['embedding'].iloc[0], str):
        df['embedding'] = df['embedding'].apply(ast.literal_eval)
    emb_matrix = np.array(df['embedding'].to_list(), dtype=float)
    os.makedirs(os.path.dirname(matrix_path), exist_ok=True)
    np.save(matrix_path, emb_matrix)
    logging.info(f"💾 Saved → {matrix_path}")
    return emb_matrix
'''
def main():
    setup_logger()
    load_dotenv()
    cfg = load_config()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    base_dir = os.path.dirname(os.path.abspath(__file__))

    movies_path = os.path.join(base_dir, cfg["data_dir"], cfg["movies_file"])
    tags_path = os.path.join(base_dir, cfg["data_dir"], cfg["tags_file"])
    embeddings_path = os.path.join(base_dir, cfg["embeddings_dir"], cfg["embeddings_file"])
    matrix_path = os.path.join(base_dir, cfg["embeddings_dir"], "embedding_matrix.npy")

    df = load_data(movies_path, tags_path)
    df = generate_embeddings(df, cfg["embedding_model"], client, embeddings_path)
    embedding_matrix(df, matrix_path)
'''