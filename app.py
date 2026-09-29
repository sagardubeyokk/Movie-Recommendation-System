import streamlit as st
import nltk
import sklearn
import pandas as pd
import joblib
import pickle

st.title("Movie Recommendation System")
with open ("movies.pickle",'rb') as m:
    movies=pickle.load(m)

similarity=joblib.load("similarity.joblib")



movie_names=movies['title'].values

# this function take the input as a movie name and recommend 5 similar movies names

def recommend(name_movie):
    
    movie_index=movies[movies['title']==name_movie].index[0]
    
    # get its similarity scores with all movies
    recommendations=similarity[movie_index]
    
    #sort high to low, skip itself, keep top 5
    movie_list=sorted(enumerate(recommendations),reverse=True,key=lambda x:x[1])[1:6]
    recommend_movies=[]
    
    # print the titles
    for i in movie_list:
        recommend_movies.append(movies.iloc[i[0]].title)
    return recommend_movies

name_movie=st.selectbox("Enter the Movie Name",movie_names)


if st.button("Recommend"):
    r=recommend(name_movie)
    st.write("Watch these movies :")

    for i in r:
        st.write(i)