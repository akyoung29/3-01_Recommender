from flask import Flask, render_template, request
import pickle
import joblib
import pandas as pd
import numpy as np
import requests  # Added for Azure endpoint call

app = Flask(__name__)

# Load the saved content filtering model
with open("models/content_filtering_model.sav", "rb") as model_file:
    vectorizer, cosine_sim, df_content = pickle.load(model_file)

# Load the original datasets
df_articles = pd.read_csv("/Users/cooperburden/Downloads/shared_articles.csv")
df_interactions = pd.read_csv("/Users/cooperburden/Downloads/users_interactions.csv")

# Merge the two DataFrames
df = pd.merge(df_articles, df_interactions, on='contentId', how='inner')

# Define the mapping for eventType
event_type_mapping = {
    'VIEW': 1,
    'LIKE': 2,
    'FOLLOW': 3,
    'BOOKMARK': 4,
    'COMMENT CREATED': 5
}

# Relabel the values in 'eventType_y'
df['eventType_y'] = df['eventType_y'].map(event_type_mapping)

# Filter articles with 14 or more interactions
interaction_counts = df['contentId'].value_counts()
keep_list = interaction_counts[interaction_counts >= 14]
df_filtered = df[df['contentId'].isin(keep_list.index)]

# Load the saved collaborative filtering model (KNN)
knn_model = joblib.load("models/knn_model.sav")

# Extract content IDs for the dropdown, limited to first 5
content_ids = df_content['contentId'].tolist()[:5]

# Azure endpoint details (replace with actual values from your group member)
AZURE_ENDPOINT_URL = "http://e8dc850d-4e76-4a7b-88b1-4e1e40ea4505.eastus2.azurecontainer.io/score"  # Replace with actual endpoint URL
AZURE_API_KEY = "TDPjqIW0IZcz81zoTMZloMQ1IMYI5e6K"  # Replace with actual API key if required

# Recommendation function for content filtering
def recommend_articles(content_id, num_recommendations=5):
    if content_id not in df_content['contentId'].values:
        return df_content[['contentId', 'title']].sample(n=min(num_recommendations, len(df_content))).values.tolist()
    idx = df_content[df_content['contentId'] == content_id].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations + 1]
    article_indices = [i[0] for i in sim_scores]
    return df_content.iloc[article_indices][['contentId', 'title']].values.tolist()

# Recommendation function for collaborative filtering (KNN)
def recommend_articles_knn(content_id, num_recommendations=5):
    if content_id in df_filtered['contentId'].values:
        content_data = df_filtered[df_filtered['contentId'] == content_id]
        if not content_data.empty:
            content_idx = content_data.index[0]
            distances, indices = knn_model.kneighbors(
                [df_filtered.iloc[content_idx][['contentId', 'authorPersonId']].values],
                n_neighbors=num_recommendations + 1
            )
            recommended_articles = []
            for idx in indices[0][1:]:
                recommended_articles.append(df_filtered.iloc[idx][['contentId', 'title']].values.tolist())
            return recommended_articles
    return df_filtered[['contentId', 'title']].sample(n=min(num_recommendations, len(df_filtered))).values.tolist()

# Recommendation function for Azure model
def recommend_articles_azure(content_id, num_recommendations=5):
    try:
        # Prepare the payload for the Azure endpoint
        payload = {"contentId": content_id, "num_recommendations": num_recommendations}
        headers = {"Authorization": f"Bearer {AZURE_API_KEY}", "Content-Type": "application/json"}

        # Make the request to the Azure endpoint
        response = requests.post(AZURE_ENDPOINT_URL, json=payload, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Assuming the response is a JSON list of [contentId, title] pairs
        recommendations = response.json()
        return recommendations[:num_recommendations]  # Limit to requested number
    except Exception as e:
        # Fallback in case of error (e.g., endpoint down or malformed response)
        print(f"Azure endpoint error: {str(e)}")
        return df_content[['contentId', 'title']].sample(n=min(num_recommendations, len(df_content))).values.tolist()

@app.route('/', methods=['GET', 'POST'])
def index():
    content_recommendations = None
    collaborative_recommendations = None
    azure_recommendations = None
    selected_content_id = None
    error = None

    if request.method == 'POST':
        selected_content_id = request.form.get('selected_content_id')
        
        if selected_content_id:
            try:
                selected_content_id = int(selected_content_id)
                # Get recommendations from all three models based on contentId
                content_recommendations = recommend_articles(selected_content_id)
                collaborative_recommendations = recommend_articles_knn(selected_content_id)
                azure_recommendations = recommend_articles_azure(selected_content_id)
            except ValueError:
                error = "Invalid Content ID. Please select a valid option."
            except Exception as e:
                error = f"An error occurred: {str(e)}"
        else:
            error = "Please select a Content ID."

    return render_template('index.html', 
                           content_ids=content_ids,
                           content_recommendations=content_recommendations,
                           collaborative_recommendations=collaborative_recommendations,
                           azure_recommendations=azure_recommendations,
                           selected_content_id=selected_content_id,
                           error=error,
                           cache_buster=str(np.random.randint(1000000)))

if __name__ == '__main__':
    app.run(debug=True)