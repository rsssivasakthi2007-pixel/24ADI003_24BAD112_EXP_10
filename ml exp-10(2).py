print("SIVASAKTHI S 24BAD112")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

ratings_path = r"C:\Users\priya\Downloads\archive (32)\ml-latest-small\ratings.csv"
movies_path = r"C:\Users\priya\Downloads\archive (32)\ml-latest-small\movies.csv"

print("========== LOADING DATA ==========")
ratings = pd.read_csv(ratings_path)
movies = pd.read_csv(movies_path)

print("Ratings shape:", ratings.shape)
print("Movies shape:", movies.shape)

user_item_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating')

user_item_filled = user_item_matrix.fillna(0)

R = user_item_filled.values

k = 20

print("\n========== APPLYING NMF ==========")
nmf_model = NMF(n_components=k, init='random', random_state=42, max_iter=200)

W = nmf_model.fit_transform(R)
H = nmf_model.components_

R_pred = np.dot(W, H)

pred_df = pd.DataFrame(R_pred, columns=user_item_matrix.columns, index=user_item_matrix.index)

def predict_rating(user, movie):
    if user in pred_df.index and movie in pred_df.columns:
        return pred_df.loc[user, movie]
    return np.nan

train, test = train_test_split(ratings, test_size=0.2, random_state=42)

y_true = []
y_pred = []

for row in test.itertuples():
    if row.userId in pred_df.index and row.movieId in pred_df.columns:
        y_true.append(row.rating)
        y_pred.append(pred_df.loc[row.userId, row.movieId])

rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print("\n========== EVALUATION ==========")
print("RMSE:", rmse)

def precision_recall_at_k(pred_df, original_df, k=10, threshold=3.5):
    precisions = []
    recalls = []
    for user in pred_df.index:
        pred_ratings = pred_df.loc[user].sort_values(ascending=False)
        top_k = pred_ratings.head(k).index
        actual_ratings = original_df.loc[user].dropna()
        relevant = actual_ratings[actual_ratings >= threshold].index
        recommended_relevant = set(top_k).intersection(set(relevant))
        precision = len(recommended_relevant) / k if k > 0 else 0
        recall = len(recommended_relevant) / len(relevant) if len(relevant) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)

    return np.mean(precisions), np.mean(recalls)
precision, recall = precision_recall_at_k(pred_df, user_item_matrix, k=10)

print("\n========== PRECISION & RECALL ==========")
print("Precision@10:", precision)
print("Recall@10:", recall)

def recommend_movies(user_id, num_recommendations=5):
    user_ratings = user_item_matrix.loc[user_id]
    already_rated = user_ratings.dropna().index

    recommendations = pred_df.loc[user_id].drop(already_rated)
    top_movies = recommendations.sort_values(ascending=False).head(num_recommendations)

    result = movies[movies['movieId'].isin(top_movies.index)][['movieId', 'title']]
    result['predicted_rating'] = top_movies.values

    return result
print("\n========== TOP RECOMMENDATIONS ==========")
rec = recommend_movies(user_id=1, num_recommendations=5)
print(rec)

print("\n========== HEATMAP: ORIGINAL ==========")
plt.figure(figsize=(10, 6))
sns.heatmap(user_item_filled.iloc[:20, :20])
plt.title("Original Matrix")
plt.xlabel("Movie ID")
plt.ylabel("User ID")
plt.show()

print("\n========== HEATMAP: RECONSTRUCTED ==========")
plt.figure(figsize=(10, 6))
sns.heatmap(pred_df.iloc[:20, :20])
plt.title("Reconstructed Matrix (NMF)")
plt.xlabel("Movie ID")
plt.ylabel("User ID")
plt.show()

print("\n========== LATENT FEATURES ==========")
plt.figure()
plt.plot(W[0])
plt.title("User Latent Features (User 1)")
plt.xlabel("Feature Index")
plt.ylabel("Weight")
plt.show()

print("\n========== RECOMMENDATION GRAPH ==========")
plt.figure()
plt.barh(rec['title'], rec['predicted_rating'], label='Predicted Rating')
plt.xlabel("Predicted Rating")
plt.ylabel("Movies")
plt.title("Top Recommended Movies")
plt.legend()
plt.gca().invert_yaxis()
plt.show()
