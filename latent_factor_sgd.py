import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data import n_items, n_users, test_df, to_indexed, train_df

# Hiperparâmetros
K = 20
EPOCAS = 20
ALPHA0 = 0.01
DECAY = 0.9
LMBDA_B = 0.02
LMBDA_F = 0.02

u_train, i_train, r_train = to_indexed(train_df)
u_test, i_test, y_true = to_indexed(test_df)

# Inicialização
rng = np.random.RandomState(42)
mu = r_train.mean()
b_u = np.zeros(n_users)
b_i = np.zeros(n_items)
P = rng.normal(0, 0.1, size=(n_users, K))
Q = rng.normal(0, 0.1, size=(n_items, K))

alpha = ALPHA0
for epoca in range(EPOCAS):
    perm = np.random.RandomState(epoca).permutation(len(r_train))
    sq_err = 0.0
    for j in perm:
        u, i, r = u_train[j], i_train[j], r_train[j]

        pred = mu + b_u[u] + b_i[i] + P[u] @ Q[i]
        e = r - pred
        sq_err += e * e

        pu = P[u].copy()
        b_u[u] += alpha * (e - LMBDA_B * b_u[u])
        b_i[i] += alpha * (e - LMBDA_B * b_i[i])
        P[u] += alpha * (e * Q[i] - LMBDA_F * P[u])
        Q[i] += alpha * (e * pu - LMBDA_F * Q[i])

    alpha *= DECAY
    train_rmse = np.sqrt(sq_err / len(r_train))
    print(f"Época {epoca + 1:2d} | lr={alpha:.5f} | train RMSE={train_rmse:.4f}")

# Avaliação no teste
y_pred = mu + b_u[u_test] + b_i[i_test] + np.sum(P[u_test] * Q[i_test], axis=1)
y_pred = np.clip(y_pred, 0.5, 5.0)

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
print(f"\nRMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")

# RMSE= 0.8822
# MAE= 0.6743
