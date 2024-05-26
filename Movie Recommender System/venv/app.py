import pickle
import streamlit as st
import requests
import pandas as pd

# Function to fetch movie poster using The Movie Database API
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500/{poster_path}"
    return "https://via.placeholder.com/500x750?text=No+Image+Available"

# Function to recommend movies based on similarity
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:6]:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_names.append(movies.iloc[i[0]].title)
        recommended_movie_posters.append(fetch_poster(movie_id))
    return recommended_movie_names, recommended_movie_posters

# Streamlit app header
st.header('Movie Recommender System')

# Load the movie data and similarity matrix
with open('movies_dict.pkl', 'rb') as f:
    movies_dict = pickle.load(f)
movies = pd.DataFrame(movies_dict)

with open('similarity.pkl', 'rb') as f:
    similarity = pickle.load(f)

# Ensure movies DataFrame has the necessary columns
if 'title' not in movies.columns or 'movie_id' not in movies.columns:
    st.error("The movies data does not contain 'title' or 'movie_id' columns.")
else:
    # Movie selection box
    movie_list = movies['title'].tolist()
    selected_movie = st.selectbox("Type or select a movie from the dropdown", movie_list)

    # Show recommendations when the button is clicked
    if st.button('Show Recommendation'):
        recommended_movie_names, recommended_movie_posters = recommend(selected_movie)
        columns = st.columns(5)
        for col, name, poster in zip(columns, recommended_movie_names, recommended_movie_posters):
            with col:
                st.text(name)
                st.image(poster)
