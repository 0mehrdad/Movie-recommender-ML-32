import streamlit as st
import requests, os
from dotenv import load_dotenv

load_dotenv()
API_URL = "http://127.0.0.1:8000"
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

st.title("🎬 Movie Recommender")

tab1, tab2 = st.tabs(["User Recommendations", "Similar Movies"])

with tab1:
    user_id = st.number_input("User ID", min_value=1, value=1)
    top_n = st.slider("Top N", 1, 20, 5)
    if st.button("Get Recommendations"):
        res = requests.get(f"{API_URL}/recommend/user/{user_id}?top_n={top_n}")
        if res.status_code == 200:
            movies = res.json()
            for m in movies:
                poster = m['poster']
                cols = st.columns([1,3])
                with cols[0]:
                    if poster:
                        st.image(poster, width=120)
                with cols[1]:
                    st.write(f"**{m['title']}**")
                    st.caption(f"Predicted Rating: {m['predicted_rating']}")
        else:
            st.error(res.json()["detail"])

with tab2:
    movie_id = st.number_input("Movie ID", min_value=1, value=1, key="movie")
    top_k = st.slider("Top K", 1, 20, 5, key="k")
    if st.button("Find Similar Movies"):
        res = requests.get(f"{API_URL}/similar/{movie_id}?top_k={top_k}")
        if res.status_code == 200:
            movies = res.json()
            for m in movies:
                poster = m['poster']
                cols = st.columns([1,3])
                with cols[0]:
                    if poster:
                        st.image(poster, width=120)
                with cols[1]:
                    st.write(f"**{m['title']}**")
                    st.caption(f"Similarity: {round(m['similarity'],3)}")
        else:
            st.error(res.json()["detail"])
