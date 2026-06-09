from collections import defaultdict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data import n_items, n_users, test_df, to_indexed, train_df

# r̂_ui = b_ui + q_i^T (p_u + |N(u)|^(-1/2) Σ_{j∈N(u)} y_j)

K = 20
EPOCAS = 20
LEARNING_RATE = 0.005
LAMBDA = 0.02
DECAY = 0.9
ITEM_BIAS_SHRINKAGE = 10
USER_BIAS_SHRINKAGE = 10
SEED = 42

rng = np.random.RandomState(SEED)

u_train, i_train, r_train = to_indexed(train_df)
u_test, i_test, r_test = to_indexed(test_df)


def build_lookups(u_arr, i_arr, r_arr):
    ratings_by_user = defaultdict(dict)  # u_idx -> {i_idx: rating}
    ratings_by_item = defaultdict(dict)  # i_idx -> {u_idx: rating}
    for u, i, r in zip(u_arr, i_arr, r_arr):
        ratings_by_user[u][i] = r
        ratings_by_item[i][u] = r
    return ratings_by_user, ratings_by_item


ratings_by_user, ratings_by_item = build_lookups(u_train, i_train, r_train)

# N(u) pré-computado como array de índices
N = {u: np.array(list(items.keys())) for u, items in ratings_by_user.items()}


def compute_baselines(ratings_by_user, ratings_by_item, global_mean):
    item_bias = defaultdict(float)
    for item, item_users in ratings_by_item.items():
        item_bias[item] = sum(r - global_mean for r in item_users.values()) / (
            ITEM_BIAS_SHRINKAGE + len(item_users)
        )
    user_bias = defaultdict(float)
    for user, user_items in ratings_by_user.items():
        user_bias[user] = sum(
            r - global_mean - item_bias[item] for item, r in user_items.items()
        ) / (USER_BIAS_SHRINKAGE + len(user_items))
    return user_bias, item_bias


global_mean = float(np.mean(r_train))
user_bias_t, item_bias_t = compute_baselines(
    ratings_by_user, ratings_by_item, global_mean
)

P = rng.normal(0, 0.1, size=(n_users, K))  # p_u
Q = rng.normal(0, 0.1, size=(n_items, K))  # q_i
Y = rng.normal(0, 0.1, size=(n_items, K))  # y_j


def predict(u, i):
    baseline = global_mean + user_bias_t.get(u, 0.0) + item_bias_t.get(i, 0.0)
    Nu = N.get(u, np.empty(0, dtype=int))
    if len(Nu) > 0:
        implicit_sum = Y[Nu].sum(axis=0) * (len(Nu) ** -0.5)
    else:
        implicit_sum = np.zeros(K)
    user_repr = P[u] + implicit_sum
    return baseline + np.dot(Q[i], user_repr)


def train():
    γ = LEARNING_RATE
    λ = LAMBDA
    for epoca in range(EPOCAS):
        order = rng.permutation(len(r_train))
        sq_err = 0.0
        for k in order:
            u, i, r = u_train[k], i_train[k], r_train[k]

            Nu = N.get(u, np.empty(0, dtype=int))
            norm = len(Nu) ** -0.5 if len(Nu) > 0 else 0.0
            implicit_sum = Y[Nu].sum(axis=0) * norm if len(Nu) > 0 else np.zeros(K)
            user_repr = P[u] + implicit_sum

            pred = (
                global_mean + user_bias_t[u] + item_bias_t[i] + np.dot(Q[i], user_repr)
            )
            e = r - pred
            sq_err += e * e

            user_bias_t[u] += γ * (e - λ * user_bias_t[u])
            item_bias_t[i] += γ * (e - λ * item_bias_t[i])

            q_i_old = Q[i].copy()
            P[u] += γ * (e * q_i_old - λ * P[u])
            Q[i] += γ * (e * user_repr - λ * q_i_old)
            if len(Nu) > 0:
                Y[Nu] += γ * (e * norm * q_i_old - λ * Y[Nu])

        rmse = np.sqrt(sq_err / len(r_train))
        print(f"época {epoca + 1:2d}: RMSE_train = {rmse:.4f}")
        γ *= DECAY


def evaluate(u_arr, i_arr, r_arr):
    preds = np.array([predict(u, i) for u, i in zip(u_arr, i_arr)])
    preds = np.clip(preds, 0.5, 5.0)
    rmse = np.sqrt(mean_squared_error(r_arr, preds))
    mae = mean_absolute_error(r_arr, preds)
    return rmse, mae


train()
rmse, mae = evaluate(u_test, i_test, r_test)
print(f"\nRMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
