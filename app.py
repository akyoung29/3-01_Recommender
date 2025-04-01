from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the saved content filtering model
with open("models/content_filtering_model.sav", "rb") as model_file:
    vectorizer, cosine_sim, df_content = pickle.load(model_file)

# Extract content IDs for the dropdown
content_ids = df_content['contentId'].tolist()

# Recommendation function
def recommend_articles(content_id, num_recommendations=5):
    if content_id not in df_content['contentId'].values:
        return []

    idx = df_content[df_content['contentId'] == content_id].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations + 1]  # Skip the first (itself)
    article_indices = [i[0] for i in sim_scores]

    return df_content.iloc[article_indices][['contentId', 'title']].values.tolist()

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = None
    selected_content_id = None
    if request.method == 'POST':
        selected_content_id = int(request.form['content_id'])
        recommendations = recommend_articles(selected_content_id)

    return render_template('index.html', content_ids=content_ids, recommendations=recommendations, content_id=selected_content_id)

if __name__ == '__main__':
    app.run(debug=True)
