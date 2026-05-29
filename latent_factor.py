from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

LMBDA_B = 0.02
ALPHA0 = 0.01
DECAY = 0.9
EPOCAS = 40

ratings = pd.read_csv("dataset/ratings.csv")
matrix_r = ratings.pivot_table(index="userId", columns="movieId", values="rating")

# Mascarar dados de treino
mask = ~np.isnan(matrix_r.values)
rows, cols = np.where(mask)
values = matrix_r.values[rows, cols]
train_idx, test_idx = train_test_split(
    range(len(values)), test_size=0.2, random_state=42
)

R_train = np.full(matrix_r.shape, np.nan)
R_train[rows[train_idx], cols[train_idx]] = values[train_idx]
print(R_train)

# Baseline: aprender mu + b_u + b_i por SGD no treino
mu = np.nanmean(R_train)
b_u = defaultdict(float)
b_i = defaultdict(float)

train_rows_user = rows[train_idx]
train_cols_item = cols[train_idx]
train_vals_ratings = values[train_idx]

for epoca in range(EPOCAS):
    perm = np.random.RandomState(epoca).permutation(len(train_vals_ratings))
    for j in perm:
        u, i, r = train_rows_user[j], train_cols_item[j], train_vals_ratings[j]
        e = r - (mu + b_u[u] + b_i[i])
        b_u[u] += ALPHA0 * (e - LMBDA_B * b_u[u])
        b_i[i] += ALPHA0 * (e - LMBDA_B * b_i[i])
    ALPHA0 *= DECAY

# Matriz de baseline b_ui (mesma shape da R)
bu_arr = np.array([b_u[u] for u in range(matrix_r.shape[0])])[:, None]
bi_arr = np.array([b_i[i] for i in range(matrix_r.shape[1])])[None, :]
B = mu + bu_arr + bi_arr  # (n_users, n_items)

# SVD sobre o RESÍDUO (R - b_ui)
R_centered = R_train - B
R_filled = np.nan_to_num(R_centered)

k = 50
sparse = csr_matrix(R_filled)
U, s, Vh = svds(sparse, k=k)
U, s, Vh = U[:, ::-1], s[::-1], Vh[::-1, :]

P = U @ np.diag(s)
Qt = Vh

# Previsao: r_ui = b_ui + p_u . q_i
R_hat = (P @ Qt) + B

y_true = values[test_idx]
y_pred = R_hat[rows[test_idx], cols[test_idx]]
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
# RMSE= 0.8732
# MAE= 0.6664
