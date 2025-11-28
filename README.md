# 🎬 Movie Recommender (Local Version: FastAPI + Streamlit + SVD + Embeddings)

This repository contains the full local version of the movie recommendation system.  
It includes:

✔ FastAPI backend  
✔ Streamlit frontend  
✔ SVD-based user recommendations  
✔ Embedding-based movie similarity  
✔ TMDB poster integration  

This is NOT the AWS version.  
This version is heavier and includes both endpoints + UI.


For the AWS version:

👉 Live App: https://movie-recommender-frontend-bx3xdouaxpshlaeiw5b4rk.streamlit.app/

👉 Frontend Repo: https://github.com/0mehrdad/movie-recommender-frontend

---

## 🚀 Features

### 1. FastAPI Backend

Exposes two endpoints:

```
GET /recommend/user/{user_id}?top_n=10
GET /similar/{movie_id}?top_k=10
```

### 2. Streamlit Frontend

- Located in `app.py`
- Calls the local API
- Displays posters, ratings, similarities

### 3. Local ML Models

- SVD trained on MovieLens
- Embeddings generated from movie tags
- Loads saved models or trains if missing

---

## 📁 Project Structure

```
Movie-recommender-ML-32/
│
├── app.py
├── config.yaml
├── requirements.txt
├── Dockerfile
├── README.md
│
├── app/
│   ├── api.py
│   ├── train_svd.py
│   ├── create_embeddings.py
│   ├── utils.py
│   └── __init__.py
│
├── notebooks/
│   ├── EDA.ipynb
│   └── models.ipynb
│
└── image_loader.ipynb
```

---

## How the System Works

### 1. SVD Model (User Recommendations)

- Trained using MovieLens ratings  
- Predicts ratings for unseen movies  
- Returned via `/recommend/user/{id}`

### 2. Embedding Model (Movie Similarity)

- Uses movie tags  
- Generates text embeddings  
- Computes cosine similarity  
- Returned via `/similar/{movie_id}`

### 3. 🖼 TMDB Poster Fetching

- Uses links.csv to map movieId -> tmdbId  
- Fetches posters with TMDB API  
- Requires TMDB_API_KEY in `.env`

### 4. 🌐 Streamlit UI

- Calls FastAPI backend  
- Shows posters, titles, scores  
- Two tabs:
  - User Recommendations
  - Similar Movies

---

## 🛠 Running Locally

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Create .env

```
TMDB_API_KEY=your_api_key_here
```

### 3. Run FastAPI backend

```
uvicorn app.api:app --reload
```

API docs:

```
http://127.0.0.1:8000/docs
```

### 4. Run Streamlit frontend

```
streamlit run app.py
```

Streamlit UI:

```
http://localhost:8501
```

---

## Difference From AWS Version

```
Local Version (this repo):
- Includes SVD user recommendations
- Includes embedding similarity
- Includes Streamlit UI
- Trains or loads heavy models locally

AWS Version:
- Only includes similarity endpoint
- No SVD (too expensive)
- Streamlit hosted separately
- Lightweight backend
```




