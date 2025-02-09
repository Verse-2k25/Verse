from flask import Flask, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# Load the trained model
with open("model.pkl", "rb") as file:
    train_data, vectorizer, similarity_matrix = pickle.load(file)

def recommend_friends(user_id, top_n=3):
    if user_id not in train_data['user_id'].values:
        return []

    user_idx = train_data[train_data['user_id'] == user_id].index[0]
    similarity_scores = similarity_matrix[user_idx]
    similar_users_indices = similarity_scores.argsort()[::-1][1:top_n+1]

    recommended_users = train_data.iloc[similar_users_indices][['user_id', 'name', 'preferences', 'artists']].to_dict(orient="records")
    return recommended_users

@app.route("/recommendations/<int:user_id>", methods=["GET"])
def get_recommendations(user_id):
    recommended_friends = recommend_friends(user_id)
    return jsonify(recommended_friends)

if __name__ == "__main__":
    app.run(debug=True)
