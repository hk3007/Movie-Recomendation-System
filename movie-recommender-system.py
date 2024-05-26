import numpy as np
import pandas as pd
import ast
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

movies.head()
credits.head()

movies = movies.merge(credits,on='title')

#geners
#id
#keywords
#title
#overview
#cast
#crew


movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


movies.dropna(inplace=True)

def convert_1(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

def convert_2(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
        else:
            break
    return L

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    for i in movie_list:
        print(new_df.iloc[i[0]].title)
        
        
ps = PorterStemmer()

movies['genres'] = movies['genres'].apply(convert_1)
movies['keywords'] = movies['keywords'].apply(convert_1)
movies['cast'] = movies['cast'].apply(convert_2)


movies['crew'] = movies['crew'].apply(fetch_director)

movies['overview'] = movies['overview'].apply(lambda x: x.split())


movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for  i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for  i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for  i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for  i in x])

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew'] 

new_df = movies[['movie_id','title','tags']]

new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))


new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())

new_df['tags'] = new_df['tags'].apply(stem)
cv = CountVectorizer(max_features=5000,stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

similarity = cosine_similarity(vectors)


recommend('Batman Begins')


pickle.dump(new_df.to_dict(), open('movies_dict.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))
