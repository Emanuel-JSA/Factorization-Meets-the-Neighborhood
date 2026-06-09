from collections import defaultdict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data import test_df, train_df

# R^k(i,u) => K vizinhos de i que o u avaliou
# N^k(i,u) =: K vizinhos de i que u interagiu implicitamente
# r̂_ui = b_ui + |R^k(i,u)|^(-1/2) · Σ_{j∈R^k(i,u)} (r_uj − b_uj) · w_ij
#                + |N^k(i,u)|^(-1/2) · Σ_{j∈N^k(i,u)} c_ij
#
# Sem feedback implícito: N(u) = R(u) (o ato de avaliar j é o sinal implícito),
# logo N^k(i,u) = R^k(i,u) e os vizinhos N(i) servem para os dois termos.

NUM_NEIGHBORS = 40
SIM_SHRINKAGE = 100
ITEM_BIAS_SHRINKAGE = 10  # shrinkage do bias de item
USER_BIAS_SHRINKAGE = 10  # shrinkage do bias de usuário
LEARNING_RATE = 0.005
LAMBDA = 0.02
NUM_EPOCHS = 30

train = list(zip(train_df.userId, train_df.movieId, train_df.rating))


# R(u) = itens que u avaliou, R(i) = usuários que avaliaram i
def build_lookups(df):
    ratings_by_user = defaultdict(dict)  # user -> {item: rating}
    ratings_by_item = defaultdict(dict)  # item -> {user: rating}
    for user, item, rating in zip(
        df.userId.values, df.movieId.values, df.rating.values
    ):
        ratings_by_user[user][item] = rating
        ratings_by_item[item][user] = rating
    return ratings_by_user, ratings_by_item


#   Correlação de Pearson entre cada par de itens, calculada sobre os
#   usuários que avaliaram os dois (o "suporte comum" U_ij).
#   ρ_ij = Σ(r_ui − r̄_i)(r_uj − r̄_j) / sqrt(Σ(r_ui − r̄_i)² · Σ(r_uj − r̄_j)²)
#   shrinkage: encolhe a correlação de pares com poucos usuários em comum.
def pearson_similarities(ratings_by_item, shrinkage=SIM_SHRINKAGE):
    items = list(ratings_by_item)
    similarities = {}
    for a in range(len(items)):
        item_i = items[a]
        for b in range(a + 1, len(items)):
            item_j = items[b]

            # usuários que avaliaram os dois itens
            common_users = (
                ratings_by_item[item_i].keys() & ratings_by_item[item_j].keys()
            )
            if len(common_users) < 2:
                continue

            # notas dos dois itens, na mesma ordem de usuários
            r_i = np.array([ratings_by_item[item_i][u] for u in common_users])
            r_j = np.array([ratings_by_item[item_j][u] for u in common_users])

            # Pearson: covariância normalizada pelos desvios
            r_i = r_i - r_i.mean()
            r_j = r_j - r_j.mean()
            denom = np.sqrt((r_i**2).sum() * (r_j**2).sum())
            if denom == 0:
                continue
            correlation = (r_i * r_j).sum() / denom

            n = len(common_users)
            similarities[(item_i, item_j)] = n / (n + shrinkage) * correlation
    return similarities


#   Top-K vizinhos por item: N(i) = K itens j com maior s_ij.
#   sim é simétrico (guardamos só o par ordenado), então alimenta os dois lados.
def topk_neighbors(similarities, k=NUM_NEIGHBORS):
    candidates_by_item = defaultdict(list)  # item -> [(sim, vizinho), ...]
    for (item_a, item_b), similarity in similarities.items():
        candidates_by_item[item_a].append((similarity, item_b))
        candidates_by_item[item_b].append((similarity, item_a))
    neighbors = {}
    for item, candidates in candidates_by_item.items():
        candidates.sort(reverse=True)
        neighbors[item] = [neighbor for _, neighbor in candidates[:k]]
    return neighbors


#   Baseline congelada b_ui = μ + b_u + b_i, estimada em forma fechada
#   Item primeiro, usuário depois sobre o resíduo (r_ui − μ − b_i).
#   O λ no denominador encolhe o bias de quem tem poucas avaliações para 0.
def compute_baselines(ratings_by_user, ratings_by_item):
    global_mean = train_df.rating.mean()
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


ratings_by_user, ratings_by_item = build_lookups(train_df)
similarities = pearson_similarities(ratings_by_item)
neighbors = topk_neighbors(similarities)
global_mean, user_bias, item_bias = compute_baselines(ratings_by_user, ratings_by_item)


# R^k(i,u) = N(i) ∩ R(u)  — interseção barata feita na hora da predição
def neighbors_rated_by(user, item):
    return [j for j in neighbors.get(item, []) if j in ratings_by_user[user]]


# usar bias aprendidas como ponto de partida
user_bias_t = dict(user_bias)
item_bias_t = dict(item_bias)


# b_ui = μ + b_u + b_i  (itens/usuários nunca vistos caem só na média global)
def baseline_t(user, item):
    return global_mean + user_bias_t.get(user, 0.0) + item_bias_t.get(item, 0.0)


explicit_weights = defaultdict(
    float
)  # (i, j) -> peso explícito w_ij (direcional, i alvo / j vizinho)
implicit_weights = defaultdict(float)  # (i, j) -> peso implícito c_ij


#   r̂_ui = b_ui + |R^k|^(-1/2) Σ (r_uj − b_uj) w_ij + |N^k|^(-1/2) Σ c_ij
#   Como N(u) = R(u), os dois somatórios percorrem o mesmo R^k(i,u).
# def predict(user, item):
#     prediction = baseline_t(user, item)
#     rated_neighbors = neighbors_rated_by(user, item)
#     if not rated_neighbors:
#         return prediction
#     norm_factor = len(rated_neighbors) ** -0.5
#     for j in rated_neighbors:
#         prediction += (
#             norm_factor
#             * (ratings_by_user[user][j] - baseline_t(user, j))
#             * explicit_weights[(item, j)]
#         )
#         prediction += norm_factor * implicit_weights[(item, j)]
#     return prediction


def predict_train(user, item):
    prediction = baseline_t(user, item)
    rated_neighbors = neighbors_rated_by(user, item)
    if not rated_neighbors:
        return prediction, rated_neighbors, 0.0
    norm_factor = len(rated_neighbors) ** -0.5
    for j in rated_neighbors:
        prediction += (
            norm_factor
            * (ratings_by_user[user][j] - baseline_t(user, j))
            * explicit_weights[(item, j)]
        )
        prediction += norm_factor * implicit_weights[(item, j)]
    return prediction, rated_neighbors, norm_factor


for epoch in range(NUM_EPOCHS):
    np.random.shuffle(train)
    for user, item_i, rating in train:
        pred, rated, norm = predict_train(user, item_i)
        erro = rating - pred

        # bias
        user_bias_t[user] = user_bias_t.get(user, 0.0) + LEARNING_RATE * (
            erro - LAMBDA * user_bias_t.get(user, 0.0)
        )
        item_bias_t[item_i] = item_bias_t.get(item_i, 0.0) + LEARNING_RATE * (
            erro - LAMBDA * item_bias_t.get(item_i, 0.0)
        )

        for item_j in rated:
            w = explicit_weights[(item_i, item_j)]
            c = implicit_weights[(item_i, item_j)]
            explicit_weights[(item_i, item_j)] += LEARNING_RATE * (
                erro * norm * (ratings_by_user[user][item_j] - baseline_t(user, item_j))
                - LAMBDA * w
            )
            implicit_weights[(item_i, item_j)] += LEARNING_RATE * (
                erro * norm - LAMBDA * c
            )

    preds = [predict_train(u, i)[0] for u, i, r in train]
    reais = [r for _, _, r in train]
    print(f"epoch {epoch}: RMSE treino = {mean_squared_error(reais, preds) ** 0.5:.4f}")


test = test_df.copy()
test["predicao"] = [
    predict_train(u, i)[0] for u, i in zip(test["userId"], test["movieId"])
]

rmse = np.sqrt(mean_squared_error(test["rating"], test["predicao"]))
mae = mean_absolute_error(test["rating"], test["predicao"])
print(f"RMSE= {rmse}")
print(f"MAE= {mae}")

# RMSE= 0.8387239091666603
# MAE= 0.6390180370312284
