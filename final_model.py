from collections import defaultdict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data import n_items, n_users, test_df, to_indexed, train_df

# Modelo final de Koren (eq. 16): soma das tres camadas, treinada conjuntamente.
#
# r̂_ui = μ + b_u + b_i
#        + q_i^T ( p_u + |N(u)|^(-1/2) Σ_{j∈N(u)} y_j )            (SVD++)
#        + |R^k(i;u)|^(-1/2) Σ_{j∈R^k(i;u)} (r_uj − b_uj) w_ij      (vizinhança explícita)
#        + |N^k(i;u)|^(-1/2) Σ_{j∈N^k(i;u)} c_ij                    (vizinhança implícita)
#
# Sem feedback implícito: N(u) = R(u) e N^k(i;u) = R^k(i;u), logo os dois
# somatórios de vizinhança percorrem os mesmos top-k vizinhos de i que u avaliou.

K = 40
NUM_NEIGHBORS = 30
NUM_EPOCHS = 45
SIM_SHRINKAGE = 200
ITEM_BIAS_SHRINKAGE = 10
USER_BIAS_SHRINKAGE = 10

# Taxas de aprendizado por grupo de parâmetro (Koren: γ1, γ2, γ3)
GAMMA_BIAS = 0.003  # γ1: b_u, b_i
GAMMA_FACTOR = 0.001  # γ2: p_u, q_i, y_j
GAMMA_WEIGHT = 0.003  # γ3: w_ij, c_ij

# Regularização por grupo de parâmetro (Koren: λ6, λ7, λ8)
LAMBDA_BIAS = 0.0015  # λ6
LAMBDA_FACTOR = 0.07  # λ7
LAMBDA_WEIGHT = 0.006  # λ8

DECAY = 0.9
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

# N(u) = R(u) pré-computado como array de índices
N = {u: np.array(list(items.keys())) for u, items in ratings_by_user.items()}


#   Baseline congelada b_ui = μ + b_u + b_i em forma fechada, usada como ponto de
#   partida; o λ no denominador encolhe o bias de quem tem poucas avaliações.
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


#   Correlação de Pearson encolhida entre cada par de itens, sobre o suporte comum.
def pearson_similarities(ratings_by_item, shrinkage=SIM_SHRINKAGE):
    items = list(ratings_by_item)
    similarities = {}
    for a in range(len(items)):
        item_i = items[a]
        for b in range(a + 1, len(items)):
            item_j = items[b]

            common_users = (
                ratings_by_item[item_i].keys() & ratings_by_item[item_j].keys()
            )
            if len(common_users) < 2:
                continue

            r_i = np.array([ratings_by_item[item_i][u] for u in common_users])
            r_j = np.array([ratings_by_item[item_j][u] for u in common_users])

            r_i = r_i - r_i.mean()
            r_j = r_j - r_j.mean()
            denom = np.sqrt((r_i**2).sum() * (r_j**2).sum())
            if denom == 0:
                continue
            correlation = (r_i * r_j).sum() / denom

            n = len(common_users)
            similarities[(item_i, item_j)] = n / (n + shrinkage) * correlation
    return similarities


#   Top-K vizinhos por item: N(i) = K itens j com maior s_ij (sim é simétrico).
def topk_neighbors(similarities, k=NUM_NEIGHBORS):
    candidates_by_item = defaultdict(list)
    for (item_a, item_b), similarity in similarities.items():
        candidates_by_item[item_a].append((similarity, item_b))
        candidates_by_item[item_b].append((similarity, item_a))
    neighbors = {}
    for item, candidates in candidates_by_item.items():
        candidates.sort(reverse=True)
        neighbors[item] = [neighbor for _, neighbor in candidates[:k]]
    return neighbors


similarities = pearson_similarities(ratings_by_item)
neighbors = topk_neighbors(similarities)


# R^k(i;u) = N(i) ∩ R(u) = N^k(i;u)  — interseção barata feita na predição
def neighbors_rated_by(user, item):
    return [j for j in neighbors.get(item, []) if j in ratings_by_user[user]]


# b_ui = μ + b_u + b_i  (usuários/itens nunca vistos caem só na média global)
def baseline_t(user, item):
    return global_mean + user_bias_t.get(user, 0.0) + item_bias_t.get(item, 0.0)


P = rng.normal(0, 0.1, size=(n_users, K))  # p_u
Q = rng.normal(0, 0.1, size=(n_items, K))  # q_i
Y = rng.normal(0, 0.1, size=(n_items, K))  # y_j

explicit_weights = defaultdict(float)  # (i, j) -> w_ij
implicit_weights = defaultdict(float)  # (i, j) -> c_ij


#   Predição completa da eq. (16). Devolve também as estruturas intermediárias
#   (Nu, norm, user_repr, vizinhos, fator de norma) para reaproveitar no treino.
def predict_full(user, item):
    pred = baseline_t(user, item)

    # SVD++: q_i^T (p_u + |N(u)|^(-1/2) Σ y_j)
    Nu = N.get(user, np.empty(0, dtype=int))
    norm = len(Nu) ** -0.5 if len(Nu) > 0 else 0.0
    implicit_sum = Y[Nu].sum(axis=0) * norm if len(Nu) > 0 else np.zeros(K)
    user_repr = P[user] + implicit_sum
    pred += np.dot(Q[item], user_repr)

    # Vizinhança: termos explícito (w_ij) e implícito (c_ij)
    rated = neighbors_rated_by(user, item)
    nf = len(rated) ** -0.5 if rated else 0.0
    for j in rated:
        pred += (
            nf
            * (ratings_by_user[user][j] - baseline_t(user, j))
            * explicit_weights[(item, j)]
        )
        pred += nf * implicit_weights[(item, j)]

    return pred, Nu, norm, user_repr, rated, nf


def predict(user, item):
    return predict_full(user, item)[0]


def train():
    γ_bias = GAMMA_BIAS
    γ_factor = GAMMA_FACTOR
    γ_weight = GAMMA_WEIGHT
    for epoca in range(NUM_EPOCHS):
        order = rng.permutation(len(r_train))
        sq_err = 0.0
        for k in order:
            u, i, r = u_train[k], i_train[k], r_train[k]

            pred, Nu, norm, user_repr, rated, nf = predict_full(u, i)
            e = r - pred
            sq_err += e * e

            # biases (γ1, λ6)
            user_bias_t[u] += γ_bias * (e - LAMBDA_BIAS * user_bias_t[u])
            item_bias_t[i] += γ_bias * (e - LAMBDA_BIAS * item_bias_t[i])

            # fatores SVD++ (γ2, λ7)
            q_i_old = Q[i].copy()
            P[u] += γ_factor * (e * q_i_old - LAMBDA_FACTOR * P[u])
            Q[i] += γ_factor * (e * user_repr - LAMBDA_FACTOR * q_i_old)
            if len(Nu) > 0:
                Y[Nu] += γ_factor * (e * norm * q_i_old - LAMBDA_FACTOR * Y[Nu])

            # pesos de vizinhança (γ3, λ8)
            for j in rated:
                w = explicit_weights[(i, j)]
                c = implicit_weights[(i, j)]
                explicit_weights[(i, j)] += γ_weight * (
                    e * nf * (ratings_by_user[u][j] - baseline_t(u, j))
                    - LAMBDA_WEIGHT * w
                )
                implicit_weights[(i, j)] += γ_weight * (e * nf - LAMBDA_WEIGHT * c)

        rmse = np.sqrt(sq_err / len(r_train))
        print(f"época {epoca + 1:2d}: RMSE_train = {rmse:.4f}")
        γ_bias *= DECAY
        γ_factor *= DECAY
        γ_weight *= DECAY


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

# RMSE= 0.8658
# MAE=  0.6601
