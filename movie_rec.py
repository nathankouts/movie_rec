# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 18:18:06 2021

@author: Nathan
"""

import pandas as pd

#load data
df = pd.read_csv('C:/Users/Nathan/Documents/Data Analysis/Movie Recommender/movie_data.csv', low_memory=False)

df.head()



df['overview'].head()


#define vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vector = TfidfVectorizer(stop_words='english')

df['overview'] = df['overview'].fillna('')
tfidf_matrix = tfidf_vector.fit_transform(df['overview'])

from sklearn.metrics.pairwise import linear_kernel
sim_matrix = linear_kernel(tfidf_matrix,tfidf_matrix)

#indices
indices = pd.Series(df.index, index = df['title']).drop_duplicates()
 
indices[:10]


#now time for the recommender function
def recommender (title,sim_scores=sim_matrix):
    idx = indices[title]
    
    sim_scores = list(enumerate(sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse= True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    
    return df['title'].iloc[movie_indices]

recommender('Next')