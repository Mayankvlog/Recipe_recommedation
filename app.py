from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import tensorflow as tf
from recipe_recommender import get_recommendations, get_nn_recommendations

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    # Get form data
    query = request.form.get('query', 'chicken curry')
    rec_method = request.form.get('rec_method', 'TF-IDF + KNN')
    num_recommendations = int(request.form.get('num_recommendations', 5))
    
    # Get recommendations based on selected method
    if rec_method == "TF-IDF + KNN":
        recommendations = get_recommendations(query, top_n=num_recommendations)
        score_label = "Similarity Score"
        score_key = "similarity_score"
    else:  # Neural Network
        recommendations = get_nn_recommendations(query, top_n=num_recommendations)
        score_label = "Confidence"
        score_key = "confidence"
    
    # Format recommendations for display
    formatted_recommendations = []
    for i, rec in enumerate(recommendations):
        formatted_rec = {
            'index': i+1,
            'name': rec['name'],
            'score_label': score_label,
            'score': rec[score_key],
            'ingredients': rec['ingredients'],
            'directions': rec['directions'],
            'cuisine': rec['cuisine']
        }
        formatted_recommendations.append(formatted_rec)
    
    return render_template(
        'results.html',
        query=query,
        recommendations=formatted_recommendations,
        score_label=score_label
    )

@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    # Get JSON data
    data = request.get_json()
    query = data.get('query', 'chicken curry')
    rec_method = data.get('rec_method', 'TF-IDF + KNN')
    num_recommendations = int(data.get('num_recommendations', 5))
    
    # Get recommendations based on selected method
    if rec_method == "TF-IDF + KNN":
        recommendations = get_recommendations(query, top_n=num_recommendations)
        score_key = "similarity_score"
    else:  # Neural Network
        recommendations = get_nn_recommendations(query, top_n=num_recommendations)
        score_key = "confidence"
    
    return jsonify({
        'query': query,
        'recommendations': recommendations,
        'score_key': score_key
    })

if __name__ == '__main__':
    app.run(debug=True)