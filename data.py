"""Fonte unica de verdade para a divisao treino/teste.

Todos os modelos importam daqui, garantindo que o RMSE/MAE seja comparavel:
mesma seed, mesmo test_size e exatamente a mesma particao de avaliacoes.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RATINGS_PATH = "dataset/ratings.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2

ratings = pd.read_csv(RATINGS_PATH)

train_df, test_df = train_test_split(
    ratings, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

user_ids = ratings.userId.unique()
item_ids = ratings.movieId.unique()
user_to_idx = {uid: k for k, uid in enumerate(user_ids)}
item_to_idx = {iid: k for k, iid in enumerate(item_ids)}
n_users = len(user_ids)
n_items = len(item_ids)


def to_indexed(df):
    u = df.userId.map(user_to_idx).values
    i = df.movieId.map(item_to_idx).values
    r = df.rating.values.astype(np.float64)
    return u, i, r


def to_dense(df):
    R = np.full((n_users, n_items), np.nan)
    u, i, r = to_indexed(df)
    R[u, i] = r
    return R
