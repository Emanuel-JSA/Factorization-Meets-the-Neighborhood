from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

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

ratings = pd.read_csv("dataset/ratings.csv")


# R(u) = itens que u avaliou, R(i) = usuários que avaliaram i
def build_lookups(df):
    ratings_by_user = defaultdict(dict)  # user -> {item: rating}
    ratings_by_item = defaultdict(dict)  # item -> {user: rating}
    for user, item, rating in zip(df.userId.values, df.movieId.values, df.rating.values):
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
            common_users = ratings_by_item[item_i].keys() & ratings_by_item[item_j].keys()
            if len(common_users) < 2:
                continue

            # notas dos dois itens, na mesma ordem de usuários
            r_i = np.array([ratings_by_item[item_i][u] for u in common_users])
            r_j = np.array([ratings_by_item[item_j][u] for u in common_users])

            # Pearson: covariância normalizada pelos desvios
            r_i = r_i - r_i.mean()
            r_j = r_j - r_j.mean()
            denom = np.sqrt((r_i ** 2).sum() * (r_j ** 2).sum())
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
similarities = pearson_similarities(ratings_by_item)
neighbors = topk_neighbors(similarities)
global_mean, user_bias, item_bias = compute_baselines(ratings_by_user, ratings_by_item)


# R^k(i,u) = N(i) ∩ R(u)  — interseção barata feita na hora da predição
def neighbors_rated_by(user, item):
    return [j for j in neighbors.get(item, []) if j in ratings_by_user[user]]


# b_ui = μ + b_u + b_i  (itens/usuários nunca vistos caem só na média global)
def baseline(user, item):
    return global_mean + user_bias.get(user, 0.0) + item_bias.get(item, 0.0)


explicit_weights = defaultdict(float)  # (i, j) -> peso explícito w_ij (direcional, i alvo / j vizinho)
implicit_weights = defaultdict(float)  # (i, j) -> peso implícito c_ij


#   r̂_ui = b_ui + |R^k|^(-1/2) Σ (r_uj − b_uj) w_ij + |N^k|^(-1/2) Σ c_ij
#   Como N(u) = R(u), os dois somatórios percorrem o mesmo R^k(i,u).
def predict(user, item):
    prediction = baseline(user, item)
    rated_neighbors = neighbors_rated_by(user, item)
    if not rated_neighbors:
        return prediction
    norm_factor = len(rated_neighbors) ** -0.5
    for j in rated_neighbors:
        prediction += norm_factor * (ratings_by_user[user][j] - baseline(user, j)) * explicit_weights[(item, j)]
        prediction += norm_factor * implicit_weights[(item, j)]
    return prediction


print("n pares de similaridade:", len(similarities))
example_item = next(iter(neighbors))
print("exemplo item", example_item, "->", neighbors[example_item][:5])
example_user = next(iter(ratings_by_user))
print(
    "R^k para (u=%s, i=%s):" % (example_user, example_item),
    neighbors_rated_by(example_user, example_item)[:8],
)
