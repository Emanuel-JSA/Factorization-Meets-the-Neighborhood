import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

ratings = pd.read_csv("dataset/ratings.csv")

# LMBDA_B = 0.02
# ALPHA0 = 0.01
# DECAY = 0.9
# EPOCAS = 40


# def baseline_puro(train, test):
#     """RMSE/MAE so com mu + b_u + b_i. Piso de comparacao."""
#     mu = train["rating"].mean()
#     b_u = defaultdict(float)
#     b_i = defaultdict(float)
#     alpha = ALPHA0
#     for epoca in range(EPOCAS):
#         for row in train.sample(frac=1, random_state=epoca).itertuples():
#             e = row.rating - (mu + b_u[row.userId] + b_i[row.movieId])
#             b_u[row.userId] += alpha * (e - LMBDA_B * b_u[row.userId])
#             b_i[row.movieId] += alpha * (e - LMBDA_B * b_i[row.movieId])
#         alpha *= DECAY
#     preds = [mu + b_u[r.userId] + b_i[r.movieId] for r in test.itertuples()]
#     return (
#         np.sqrt(mean_squared_error(test["rating"], preds)),
#         mean_absolute_error(test["rating"], preds),
#     )

# rui = bui + puTqi

# Dados e montar matriz R
ratings = pd.read_csv("dataset/ratings.csv")
matrix_r = ratings.pivot_table(index="userId", columns="movieId", values="rating")
matrix_r.head()

# Mascarar dados de treino
mask = ~np.isnan(matrix_r.values)
rows, cols = np.where(mask)
values = matrix_r.values[rows, cols]
train_idx, test_idx = train_test_split(
    range(len(values)), test_size=0.2, random_state=42
)
R_train = np.full(matrix_r.shape, np.nan)
R_train[rows[train_idx], cols[train_idx]] = values[train_idx]

# Usa media geral em vez de zero quando não se tem nota
mu = np.nanmean(R_train)
R_centered = R_train - mu
R_filled = np.nan_to_num(R_centered)

k = 50
sparse = csr_matrix(np.nan_to_num(R_filled))
U, s, Vh = svds(sparse, k=k)

U = U[:, ::-1]
s = s[::-1]
Vh = Vh[::-1, :]

# P e QT
P = U @ np.diag(s)  # (n_users, k)
Qt = Vh  # (k, n_items)

# Previsao
R_hat = (P @ Qt) + mu

R_hat_df = pd.DataFrame(R_hat, index=matrix_r.index, columns=matrix_r.columns)

R_hat_df.head()

y_true = values[test_idx]
y_pred = R_hat[rows[test_idx], cols[test_idx]]

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)

print(f"\nRMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
# RMSE: 1.0112
# MAE: 0.7957
