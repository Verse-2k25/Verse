import pymongo
import pandas as pd
from flask import Flask, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

app = Flask(__name__)
CORS(app)

client = pymongo.MongoClient("mongodb+srv://debadityasom22:SpectreKnight@musicdata.jej4j.mongodb.net/")
db = client["music_genre_recommender"]
collection = db["user_data"]  

users_data = list(collection.find({}, {"_id": 0, "user_id": 1, "name": 1, "preferences": 1, "artists": 1}))

train_data = pd.DataFrame(users_data)

train_data['combined'] = train_data['preferences'].apply(lambda x: ','.join(x)) + ',' + train_data['artists'].apply(lambda x: ','.join(x))


vectorizer = TfidfVectorizer()
feature_matrix = vectorizer.fit_transform(train_data['combined'])
similarity_matrix = cosine_similarity(feature_matrix)

def recommend(user_id, train_data, similarity_matrix, top_n=3):
    if user_id not in train_data['user_id'].values:
        print(f"User ID {user_id} not found in the dataset.")
        return pd.DataFrame()
    
    user_idx = train_data[train_data['user_id'] == user_id].index[0]
    similarity_scores = similarity_matrix[user_idx]
    similar_users_indices = similarity_scores.argsort()[::-1][1:top_n+1]
    
    recommended_users = train_data.iloc[similar_users_indices]
    return recommended_users

try:
    user_id = int(input("Enter the user ID for recommendations (e.g., 1 to 20): "))
    recommendations = recommend(user_id, train_data, similarity_matrix, top_n=3)
    
    if not recommendations.empty:
        print(f"\nRecommendations for User ID {user_id}:")
        print(recommendations[['user_id', 'name', 'preferences', 'artists']])

    else:
        print("No recommendations could be made.")
except ValueError:
    print("Invalid input. Please enter a valid user ID (integer).")


with open("model.pkl", "wb") as file:
    pickle.dump((train_data, vectorizer, similarity_matrix), file)

print("Model trained and saved as model.pkl")