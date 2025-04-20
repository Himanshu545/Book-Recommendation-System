from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn import neighbors
from sklearn.preprocessing import MinMaxScaler
from fuzzywuzzy import process

app = Flask(__name__)
CORS(app)  # This enables CORS for all routes

# Load dataset
df = pd.read_csv(r'C:\Users\admin\Desktop\ML project\ML PROJECT - Copy\books.csv', on_bad_lines='skip')

# Preprocess Data
df2 = df.copy()
df2['title_lower'] = df2['title'].str.strip().str.lower()

# Prepare Features
rating = pd.get_dummies(df2['average_rating'])
lang = pd.get_dummies(df2['language_code'])
features = pd.concat([rating, lang, df2[['average_rating', 'ratings_count']]], axis=1)

# Convert column names to string format
features.columns = features.columns.astype(str)

# Normalize
scaler = MinMaxScaler()
features[['average_rating', 'ratings_count']] = scaler.fit_transform(features[['average_rating', 'ratings_count']])
features = scaler.fit_transform(features)

# Train Model
model = neighbors.NearestNeighbors(n_neighbors=6, algorithm='ball_tree')
model.fit(features)
dist, idlist = model.kneighbors(features)

# Recommendation Function
def BookRecommender(book_name):
    book_name = book_name.strip().lower()
    match = process.extractOne(book_name, df2['title_lower'])
    if not match or match[1] < 80:
        return ["Book not found"]

    book_id = df2[df2['title_lower'] == match[0]].index[0]
    similar_books = idlist[book_id][1:]
    return df2.iloc[similar_books]['title'].tolist()

# API Endpoint
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    book_name = data.get("book_name", "")
    recommendations = BookRecommender(book_name)
    return jsonify({"recommendations": recommendations})

if __name__ == '__main__':
    app.run(debug=True)
