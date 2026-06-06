import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

st.set_page_config(page_title="🎯 Recommendation System", layout="wide")
st.title("🎯 AI Recommendation System")
st.markdown("Personalized recommendations using collaborative and content-based filtering")

@st.cache_data
def load_movie_data():
    movies = pd.DataFrame({
        "movie_id": range(1, 21),
        "title": ["Inception","The Dark Knight","Interstellar","Avengers","Iron Man",
                  "The Matrix","Parasite","Joker","1917","Dunkirk",
                  "The Godfather","Pulp Fiction","Fight Club","Forrest Gump","Shawshank",
                  "The Silence","Get Out","Us","Hereditary","Midsommar"],
        "genre": ["Sci-Fi","Action","Sci-Fi","Action","Action",
                  "Sci-Fi","Thriller","Drama","War","War",
                  "Crime","Crime","Thriller","Drama","Drama",
                  "Thriller","Horror","Horror","Horror","Horror"],
    })
    np.random.seed(42)
    n_users, n_movies = 50, 20
    ratings = np.random.choice([0,1,2,3,4,5], size=(n_users, n_movies), p=[0.6,0.05,0.05,0.1,0.1,0.1])
    ratings_df = pd.DataFrame(ratings, columns=[f"m{i}" for i in range(1,21)])
    ratings_df.index = [f"user_{i}" for i in range(1, n_users+1)]
    return movies, ratings_df

movies, ratings_df = load_movie_data()

tab1, tab2 = st.tabs(["🤝 Collaborative Filtering", "📋 Content-Based"])

with tab1:
    st.subheader("User-Based Collaborative Filtering")
    user = st.selectbox("Select User:", ratings_df.index[:10])
    n_recs = st.slider("Number of recommendations:", 3, 10, 5)
    
    if st.button("🎬 Get Recommendations (Collaborative)"):
        user_sim = cosine_similarity(ratings_df)
        user_idx = list(ratings_df.index).index(user)
        sim_scores = list(enumerate(user_sim[user_idx]))
        sim_users = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
        
        user_ratings = ratings_df.loc[user]
        unwatched = user_ratings[user_ratings == 0].index
        rec_scores = {}
        for sim_user_idx, sim_score in sim_users:
            sim_user_ratings = ratings_df.iloc[sim_user_idx]
            for movie_col in unwatched:
                if sim_user_ratings[movie_col] > 3:
                    rec_scores[movie_col] = rec_scores.get(movie_col, 0) + sim_score * sim_user_ratings[movie_col]
        
        top_recs = sorted(rec_scores.items(), key=lambda x: x[1], reverse=True)[:n_recs]
        st.markdown("### 🎬 Recommended for you:")
        for i, (col, score) in enumerate(top_recs):
            movie_id = int(col.replace("m",""))
            title = movies[movies.movie_id == movie_id]["title"].values
            genre = movies[movies.movie_id == movie_id]["genre"].values
            if len(title):
                st.write(f"{i+1}. **{title[0]}** ({genre[0]}) — Score: {score:.1f}")

with tab2:
    st.subheader("Content-Based Filtering")
    movie_title = st.selectbox("I liked this movie:", movies["title"])
    
    if st.button("🎬 Find Similar Movies"):
        movie_genres = pd.get_dummies(movies["genre"])
        sim_matrix = cosine_similarity(movie_genres)
        idx = movies[movies.title == movie_title].index[0]
        sim_scores = list(enumerate(sim_matrix[idx]))
        sim_movies = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
        
        st.markdown("### 🎬 Similar movies:")
        for i, (movie_idx, score) in enumerate(sim_movies):
            st.write(f"{i+1}. **{movies.iloc[movie_idx].title}** ({movies.iloc[movie_idx].genre}) — Similarity: {score:.2f}")
