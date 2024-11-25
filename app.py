from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load TF-IDF vectorizer and cosine similarity matrix
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
cosine_sim = np.load('models/cosine_similarity.npy')

# Load the dataset
df = pd.read_csv('models/preprocessed_dataset.csv')

# Serve the HTML file (renamed index1.html to index.html)
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')  # Changed to 'index.html'

# Serve static files (e.g., JavaScript, images)
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

# Handle recommendation requests from frontend
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    product_name = data.get('product_name')

    if not product_name:
        return jsonify({'error': 'Product name not provided'}), 400

    recommendations = get_product_recommendations(product_name)
    
    if not recommendations:
        return jsonify({'error': 'No recommendations found'}), 404
    
    return jsonify({'recommendations': recommendations})

# Function to get product recommendations
def get_product_recommendations(product_name, num_recommendations=10):
    # Perform a partial match search for product names
    matching_products = df[df['product_name'].str.contains(product_name, case=False)]

    if matching_products.empty:
        return None

    # Get cosine similarity scores for the matching products
    similarity_scores = cosine_sim[matching_products.index]

    # Sum the similarity scores across products
    similarity_scores_sum = np.sum(similarity_scores, axis=0)

    # Get indices of the most similar products
    top_indices = np.argsort(similarity_scores_sum)[-num_recommendations:][::-1]

    # Retrieve unique recommendations with details
    recommendations = []
    seen_product_ids = set()
    for idx in top_indices:
        product_id = df.iloc[idx]['product_id']
        if product_id not in seen_product_ids:
            recommendation = {
                'product_id': product_id,
                'product_name': df.iloc[idx]['product_name'],
                'img_link': df.iloc[idx]['img_link'],
                'product_link': df.iloc[idx]['product_link']
            }
            recommendations.append(recommendation)
            seen_product_ids.add(product_id)

    return recommendations

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)  # Set to run on 0.0.0.0 for Render
