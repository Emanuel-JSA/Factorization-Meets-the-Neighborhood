from collections import defaultdict

import numpy as np
import pandas as pd

K = 20
DECAY = 0.9

ratings = pd.read_csv("dataset/ratings.csv")

user_ids = ratings.userId.unique()
item_ids = ratings.movieId.unique()
user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
item_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}
n_users = len(user_ids)
n_items = len(item_ids)


def build_lookups(df):
    ratings_by_user = defaultdict(dict)  # u_idx -> {i_idx: rating}
    ratings_by_item = defaultdict(dict)  # i_idx -> {u_idx: rating}
    for user, item, rating in zip(
        df.userId.values, df.movieId.values, df.rating.values
    ):
        u = user_to_idx[user]
        i = item_to_idx[item]
        ratings_by_user[u][i] = rating
        ratings_by_item[i][u] = rating
    return ratings_by_user, ratings_by_item


ratings_by_user, ratings_by_item = build_lookups(ratings)

global_mean = float(np.mean(ratings.rating.values))
user_bias_t = defaultdict(float)  # u_idx -> b_u
item_bias_t = defaultdict(float)  # i_idx -> b_i


def baseline_t(u, i):
    return global_mean + user_bias_t.get(u, 0.0) + item_bias_t.get(i, 0.0)


N = {u: np.array(list(items.keys())) for u, items in ratings_by_user.items()}

rng = np.random.RandomState(42)
P = rng.normal(0, 0.1, size=(n_users, K))
Q = rng.normal(0, 0.1, size=(n_items, K))
Y = rng.normal(0, 0.1, size=(n_items, K))


# r̂_ui = b_ui + q_i^T (p_u + |N(u)|^(-1/2) Σ_{j∈N(u)} y_j)
def predict(u, i):
    baseline = baseline_t(u, i)
    Nu = N.get(u, np.empty(0, dtype=int))
    if len(Nu) > 0:
        implicit_sum = Y[Nu].sum(axis=0) * (len(Nu) ** -0.5)
    else:
        implicit_sum = np.zeros(K)
    user_repr = P[u] + implicit_sum
    return baseline + np.dot(Q[i], user_repr)


LEARNING_RATE = 0.005
LAMBDA = 0.02
EPOCAS = 20


def train(u_train, i_train, r_train):
    γ = LEARNING_RATE
    λ = LAMBDA

    for epoca in range(EPOCAS):
        order = rng.permutation(len(r_train))
        sq_err = 0.0

        for idx in order:
            u = u_train[idx]
            i = i_train[idx]
            r = r_train[idx]

            Nu = N.get(u, np.empty(0, dtype=int))
            norm = len(Nu) ** -0.5 if len(Nu) > 0 else 0.0

            implicit_sum = Y[Nu].sum(axis=0) * norm if len(Nu) > 0 else np.zeros(K)
            user_repr = P[u] + implicit_sum

            pred = (
                global_mean + user_bias_t[u] + item_bias_t[i] + np.dot(Q[i], user_repr)
            )
            e = r - pred
            sq_err += e * e

            bu = user_bias_t[u]
            bi = item_bias_t[i]
            user_bias_t[u] += γ * (e - λ * bu)
            item_bias_t[i] += γ * (e - λ * bi)

            q_i_old = Q[i].copy()
            P[u] += γ * (e * q_i_old - λ * P[u])
            Q[i] += γ * (e * user_repr - λ * q_i_old)

            if len(Nu) > 0:
                grad_y = γ * (e * norm * q_i_old - λ * Y[Nu])
                Y[Nu] += grad_y

        rmse = np.sqrt(sq_err / len(r_train))
        γ *= DECAY
        print(f"época {epoca + 1}: RMSE_train = {rmse:.4f}")
