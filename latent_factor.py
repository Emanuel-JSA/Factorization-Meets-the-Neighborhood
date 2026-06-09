from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data import n_items, n_users, test_df, to_dense, to_indexed, train_df

LMBDA_B = 0.02
ALPHA0 = 0.01
DECAY = 0.9
EPOCAS = 40

R_train = to_dense(train_df)
print(R_train)

# Baseline: aprender mu + b_u + b_i por SGD no treino
mu = np.nanmean(R_train)
b_u = defaultdict(float)
b_i = defaultdict(float)

train_rows_user, train_cols_item, train_vals_ratings = to_indexed(train_df)

for epoca in range(EPOCAS):
    perm = np.random.RandomState(epoca).permutation(len(train_vals_ratings))
    for j in perm:
        u, i, r = train_rows_user[j], train_cols_item[j], train_vals_ratings[j]
        e = r - (mu + b_u[u] + b_i[i])
        b_u[u] += ALPHA0 * (e - LMBDA_B * b_u[u])
        b_i[i] += ALPHA0 * (e - LMBDA_B * b_i[i])
    ALPHA0 *= DECAY

# Matriz de baseline b_ui (mesma shape da R)
bu_arr = np.array([b_u[u] for u in range(n_users)])[:, None]
bi_arr = np.array([b_i[i] for i in range(n_items)])[None, :]
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

u_test, i_test, y_true = to_indexed(test_df)
y_pred = R_hat[u_test, i_test]
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
# RMSE= 0.8732
# MAE= 0.6664
