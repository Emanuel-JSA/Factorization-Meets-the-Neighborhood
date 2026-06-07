import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# r̂_ui = b_ui + q_i^T (p_u + |N(u)|^(-1/2) Σ_{j∈N(u)} y_j )
# q_i => vetor latente do item
# p_u => vetor latente do usuario
# N(u) => conjunto de itens para os quais u deu feedback
# y_j => vetor latente do item j
# Σ_{j∈N(u)} y_j => soma dos vetores latentes dos items que u interagiu
# N(u)|^(-1/2) => fator de normalização
ITEM_BIAS_SHRINKAGE = 10  # shrinkage do bias de item
USER_BIAS_SHRINKAGE = 10  # shrinkage do bias de usuário
K = 20
EPOCAS = 20
ALPHA0 = 0.01
DECAY = 0.9
LMBDA_F = 0.02
LEARNING_RATE = 0.005
LAMBDA = 0.02

ratings = pd.read_csv("dataset/ratings.csv")


def build_lookups(df):
    ratings_by_user = defaultdict(dict)  # user -> {item: rating}
    ratings_by_item = defaultdict(dict)  # item -> {user: rating}
    for user, item, rating in zip(
        df.userId.values, df.movieId.values, df.rating.values
    ):
        ratings_by_user[user][item] = rating
        ratings_by_item[item][user] = rating
    return ratings_by_user, ratings_by_item


def compute_baselines(ratings_by_user, ratings_by_item):
    global_mean = np.mean(ratings.rating.values)
    item_bias = {}  # item -> b_i
    for item, item_users in ratings_by_item.items():
        item_bias[item] = sum(r - global_mean for r in item_users.values()) / (
            ITEM_BIAS_SHRINKAGE + len(item_users)
        )
    user_bias = {}  # user -> b_u
    for user, user_items in ratings_by_user.items():
        user_bias[user] = sum(
            r - global_mean - item_bias[item] for item, r in user_items.items()
        ) / (USER_BIAS_SHRINKAGE + len(user_items))
    return global_mean, user_bias, item_bias


ratings_by_user, ratings_by_item = build_lookups(ratings)
global_mean, user_bias, item_bias = compute_baselines(ratings_by_user, ratings_by_item)

# usar bias aprendidas como ponto de partida
user_bias_t = dict(user_bias)
item_bias_t = dict(item_bias)


# b_ui = μ + b_u + b_i  (itens/usuários nunca vistos caem só na média global)
def baseline_t(user, item):
    return global_mean + user_bias_t.get(user, 0.0) + item_bias_t.get(item, 0.0)


matrix_r = ratings.pivot_table(index="userId", columns="movieId", values="rating")
n_users, n_items = matrix_r.shape

mask = ~np.isnan(matrix_r.values)
rows, cols = np.where(mask)
values = matrix_r.values[rows, cols]

train_idx, test_idx = train_test_split(
    range(len(values)), test_size=0.2, random_state=42
)

u_train = rows[train_idx]
i_train = cols[train_idx]
r_train = values[train_idx]

rng = np.random.RandomState(42)
P = rng.normal(0, 0.1, size=(n_users, K))
Q = rng.normal(0, 0.1, size=(n_items, K))

# r̂_ui = b_ui + q_i^T (p_u + |N(u)|^(-1/2) Σ_{j∈N(u)} y_j )
def predict_train(user, item_i):
    baseline = baseline_t(user, item_i)
    rated_by_u = rated_by(user)

    implicit_sum = 0.0
    for item_j in rated_by_u:
        implicit_sum += Y[item_j]

    if len(rated_by_u) > 0:
        implicit_sum *= len(rated_by_u) ** -0.5

    user_repr = P[user] + implicit_sum
    prediction = baseline + dot(Q[item_i], user_repr)
    return prediction
