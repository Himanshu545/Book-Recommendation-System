import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn.preprocessing import MinMaxScaler
from fuzzywuzzy import process  # For fuzzy matching

# Load dataset
df = pd.read_csv(r'C:\Users\admin\Desktop\ML project\ML PROJECT - Copy\books.csv', on_bad_lines='skip')

# Copy dataset for processing
df2 = df.copy()

# Categorize 'average_rating' into ranges
df2.loc[(df2['average_rating'] >= 0) & (df2['average_rating'] <= 1), 'rating_between'] = "between 0 and 1"
df2.loc[(df2['average_rating'] > 1) & (df2['average_rating'] <= 2), 'rating_between'] = "between 1 and 2"
df2.loc[(df2['average_rating'] > 2) & (df2['average_rating'] <= 3), 'rating_between'] = "between 2 and 3"
df2.loc[(df2['average_rating'] > 3) & (df2['average_rating'] <= 4), 'rating_between'] = "between 3 and 4"
df2.loc[(df2['average_rating'] > 4) & (df2['average_rating'] <= 5), 'rating_between'] = "between 4 and 5"

# One-hot encoding for categorical variables
rating = pd.get_dummies(df2['rating_between'])
lang = pd.get_dummies(df2['language_code'])

# Create features dataset
features = pd.concat([rating, lang, df2['average_rating'], df2['ratings_count']], axis=1)

# Normalize numerical features
scaler = MinMaxScaler()
features[['average_rating', 'ratings_count']] = scaler.fit_transform(features[['average_rating', 'ratings_count']])

# Convert to NumPy array
features = scaler.fit_transform(features)

# Train KNN model
model = neighbors.NearestNeighbors(n_neighbors=6, algorithm='ball_tree')
model.fit(features)

# Find nearest neighbors
dist, idlist = model.kneighbors(features)

# Book recommendation function with fuzzy matching
def BookRecommender(book_name):
    # Convert input to lowercase and remove spaces
    book_name = book_name.strip().lower()
    
    # Create a lowercase title column for comparison
    df2['title_lower'] = df2['title'].str.strip().str.lower()

    # Find closest match
    match = process.extractOne(book_name, df2['title_lower'])

    if match:
        closest_match, score = match[0], match[1]
    else:
        return ["Book not found in dataset"]

    # If the match confidence is too low, return not found
    if score < 80:
        return ["Book not found in dataset"]

    # Get book index
    book_id = df2[df2['title_lower'] == closest_match].index[0]

    # Get similar book indices (excluding itself)
    similar_books = idlist[book_id][1:]

    # Retrieve book titles
    book_list_name = df2.iloc[similar_books]['title'].tolist()
    
    return book_list_name

# Example usage
if __name__ == "__main__":
    book_name = "Harry Potter and the Half-Blood Prince (Harry Potter #6)"
    recommendations = BookRecommender(book_name)
    
    print("\n📚 Recommended Books:")
    for idx, rec_book in enumerate(recommendations, start=1):
        print(f"{idx}. {rec_book}")
