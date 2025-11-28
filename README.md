🎬 Movie Recommender — Local Full Version (FastAPI + Streamlit + SVD + Embeddings)

This repository contains the complete local version of the movie recommendation system.
It includes both:

✔ Backend — FastAPI
✔ Frontend — Streamlit UI
✔ ML Models — SVD & Embeddings
✔ Poster Fetching via TMDB API

This version runs entirely on your local machine and exposes two recommendation endpoints.

🚀 Features
🔹 FastAPI Backend

Provides 2 full endpoints:

User-based Recommendations (SVD)

GET /recommend/user/{user_id}?top_n=10


Movie-to-Movie Similarity (Embeddings)

GET /similar/{movie_id}?top_k=10

🔹 Streamlit Frontend (Local Only)

Built into this same repo (app.py)

Calls the local FastAPI backend

Shows posters + scores

Lets users test both recommenders interactively

🔹 Local ML Models

SVD model trained on MovieLens ratings

Embedding matrix generated from movie tags

Automated loading or training based on saved files

📁 Project Structure (Correct)
Movie-recommender-ML-32/
│
├── app.py                     # Streamlit frontend (local UI)
├── config.yaml                # Config for model/data paths
├── requirements.txt
├── Dockerfile
├── README.md
│
├── app/
│   ├── api.py                 # FastAPI backend with 2 endpoints
│   ├── train_svd.py           # SVD training + loading
│   ├── create_embeddings.py   # Embedding creation & loading
│   ├── utils.py               # Poster fetching, helpers
│   └── __init__.py
│
├── notebooks/
│   ├── EDA.ipynb
│   └── models.ipynb
│
└── image_loader.ipynb

🔌 How It Works
🧠 1. SVD Model (User Recommendations)

Uses MovieLens ratings

Trains/loads SVD model

Predicts unseen movie ratings

Returned via /recommend/user/{id}

🤖 2. Embedding Model (Movie Similarity)

Uses movie tags

Generates embeddings

Computes cosine similarity

Returned via /similar/{movie_id}

🖼 3. Posters from TMDB

Uses links.csv → tmdbId → poster_path

Fetched dynamically using TMDB_API_KEY

🌐 4. Streamlit Frontend

Located in app.py, it:

Calls the FastAPI backend running on http://127.0.0.1:8000

Shows posters, scores, similarity

Provides UI tabs:

User Recommendations

Similar Movies

🛠 Run Locally
1. Install dependencies
pip install -r requirements.txt

2. Add .env file
TMDB_API_KEY=your_tmdb_key_here

3. Run backend
uvicorn app.api:app --reload


API docs → http://127.0.0.1:8000/docs

4. Run Streamlit
streamlit run app.py


UI available on → http://localhost:8501
