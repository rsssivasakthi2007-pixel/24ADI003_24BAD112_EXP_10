print("SIVASAKTHI S 24BAD112")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from math import sqrt
import os

ratings_path = r"C:\Users\priya\Downloads\archive (32)\ml-latest-small\ratings.csv"
movies_path = r"C:\Users\priya\Downloads\archive (32)\ml-latest-small\movies.csv"

print("========== FILE CHECK ==========")
print("Ratings exists:", os.path.exists(ratings_path))
print("Movies exists:", os.path.exists(movies_path))

ratings = pd.read_csv(ratings_path)
movies = pd.read_csv(movies_path)

ratings = ratings[['userId', 'movieId', 'rating']]
ratings.columns = ['user_id', 'movie_id', 'rating']

train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

user_item_matrix = train_data.pivot(index='user_id', columns='movie_id', values='rating')

user_means = user_item_matrix.mean(axis=1)
normalized_matrix = user_item_matrix.sub(user_means, axis=0).fillna(0)

U, sigma, Vt = np.linalg.svd(normalized_matrix, full_matrices=False)
sigma = np.diag(sigma)

k = 50

U_k = U[:, :k]
sigma_k = sigma[:k, :k]
Vt_k = Vt[:k, :]

reconstructed_matrix = np.dot(np.dot(U_k, sigma_k), Vt_k)
reconstructed_matrix = reconstructed_matrix + user_means.values.reshape(-1, 1)

predicted_df = pd.DataFrame(reconstructed_matrix,
                            index=user_item_matrix.index,
                            columns=user_item_matrix.columns)

def predict_rating(user_id, movie_id):
    if user_id in predicted_df.index and movie_id in predicted_df.columns:
        return predicted_df.loc[user_id, movie_id]
    return np.nan

def recommend_movies(user_id, n=5):
    user_ratings = user_item_matrix.loc[user_id]
    unrated_movies = user_ratings[user_ratings.isna()].index
    predictions = predicted_df.loc[user_id, unrated_movies]
    top_movies = predictions.sort_values(ascending=False).head(n)
    result = movies[movies['movieId'].isin(top_movies.index)][['movieId', 'title']]
    result['predicted_rating'] = top_movies.values
    return result.sort_values(by='predicted_rating', ascending=False)

print("\n========== TOP RECOMMENDATIONS ==========")
rec = recommend_movies(1, 5)
print(rec)

y_true = []
y_pred = []

for row in test_data.itertuples():
    pred = predict_rating(row.user_id, row.movie_id)
    if not np.isnan(pred):
        y_true.append(row.rating)
        y_pred.append(pred)

rmse = sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)

print("\n========== MODEL EVALUATION ==========")
print("RMSE:", rmse)
print("MAE:", mae)

print("\n========== HEATMAP: ORIGINAL MATRIX ==========")
plt.figure()
sns.heatmap(user_item_matrix.fillna(0).iloc[:20, :20])
plt.title("Original User-Item Matrix")
plt.xlabel("Movie ID")
plt.ylabel("User ID")
plt.show()

print("\n========== HEATMAP: RECONSTRUCTED MATRIX ==========")
plt.figure()
sns.heatmap(predicted_df.iloc[:20, :20])
plt.title("Reconstructed Matrix using SVD")
plt.xlabel("Movie ID")
plt.ylabel("User ID")
plt.show()

k_values = [10, 20, 30, 40, 50]
rmse_list = []

for k in k_values:
    U_k = U[:, :k]
    sigma_k = sigma[:k, :k]
    Vt_k = Vt[:k, :]

    recon = np.dot(np.dot(U_k, sigma_k), Vt_k)
    recon = recon + user_means.values.reshape(-1, 1)

    pred_df = pd.DataFrame(recon,
                           index=user_item_matrix.index,
                           columns=user_item_matrix.columns)

    y_true_temp = []
    y_pred_temp = []

    for row in test_data.itertuples():
        if row.user_id in pred_df.index and row.movie_id in pred_df.columns:
            y_true_temp.append(row.rating)
            y_pred_temp.append(pred_df.loc[row.user_id, row.movie_id])

    rmse_list.append(sqrt(mean_squared_error(y_true_temp, y_pred_temp)))

print("\n========== GRAPH: ERROR vs LATENT FACTORS ==========")
plt.figure()
plt.plot(k_values, rmse_list, marker='o', label='RMSE')
plt.title("Error vs Number of Latent Factors (k)")
plt.xlabel("Latent Factors (k)")
plt.ylabel("RMSE")
plt.legend()
plt.grid()
plt.show()

print("\n========== GRAPH: TOP RECOMMENDED MOVIES ==========")
plt.figure()
plt.barh(rec['title'], rec['predicted_rating'], label='Predicted Rating')
plt.xlabel("Predicted Rating")
plt.ylabel("Movies")
plt.title("Top Recommended Movies for User 1")
plt.legend()
plt.gca().invert_yaxis()
plt.show()
