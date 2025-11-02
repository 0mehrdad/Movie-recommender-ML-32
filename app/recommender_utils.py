from train_svd import svd_model, load_data
from create_embeddings import create_embeddings, build_embedding_matrix
import os
import numpy as np
import time
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



def recommend_svd (df , model , user_id):
    movies = df['title'].unique()
    movie_watched = df[df['userId'] == user_id].title.values
    new_movies = [movie for movie in movies if movie not in movie_watched]
    predictions = {}
    for movie in new_movies:
        predict = model.predict (user_id , movie).est
        predictions [movie] = predict.round(2)
    sorted_movies = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    return sorted_movies

def similar_movies (movie_id , df, num_movies=10):
    # Find movie index
    print(df.index[df['movieId'] == movie_id])
    idx = df.index[df['movieId'] == movie_id][0]
    embeddings = create_embeddings ()
    start_time = time.time()
    emb_matrix = build_embedding_matrix (embeddings)
    print ("Embedding matrix built in %.2f seconds" % (time.time() - start_time))
    # Compute cosine similarities
    target = emb_matrix[idx]
    norms = np.linalg.norm(emb_matrix, axis=1) * np.linalg.norm(target)
    sims = np.dot(emb_matrix, target) / norms
    
    # Get top-N (num_movies) similar movies (excluding itself)
    top_idx = np.argsort(sims)[::-1][1:num_movies+1]
    return df.iloc[top_idx][['movieId', 'title' , 'text_features']]

def recommend_content_based (df , movie_id ,user_id, num_movies=10):
    watched_movies = df[df['userId'] == user_id]['movieId'].values
    print ("User has already watched movies with IDs: ", watched_movies)
    similar = similar_movies (movie_id , df , num_movies + len(watched_movies))
    print ("Similar movies found: ", similar['movieId'].values)
    recommendations = similar[~similar['movieId'].isin(watched_movies)].head(num_movies)
    return recommendations

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ratings_path = os.path.join(base_dir, CONFIG["data_dir"], CONFIG["ratings_file"])
    movies_path =os.path.join(base_dir, CONFIG["data_dir"], CONFIG["movies_file"])
           
    df = load_data(movies_path, ratings_path)
    while True:
        try :    
            movie_id = int (input ("Enter a movie ID you like: "))
            user_id = int (input ("Enter your user ID: "))
            print (recommend_content_based (df , movie_id , user_id, num_movies=5))
        except Exception as e:
            print ("Error: ", e)