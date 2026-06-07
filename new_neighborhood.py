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


#   Estatísticas de co-avaliação, acumuladas varrendo cada usuário uma vez.
#   Para cada par (i, j) avaliado pelo mesmo usuário acumulamos as somas
#   necessárias para a correlação de Pearson sobre o suporte comum U_ij.
#   Custo ~ Σ_u |R(u)|^2  (esparso, evita o item×item denso).
def cooccurrence_stats(ratings_by_user):
    pair_count = defaultdict(int)  # |U_ij|  nº de usuários que avaliaram i e j
    sum_product = defaultdict(float)  # Σ r_ui r_uj
    sum_i = defaultdict(float)  # Σ r_ui
    sum_j = defaultdict(float)  # Σ r_uj
    sum_sq_i = defaultdict(float)  # Σ r_ui^2
    sum_sq_j = defaultdict(float)  # Σ r_uj^2
    for user_ratings in ratings_by_user.values():
        rated_items = list(user_ratings.items())
        for a in range(len(rated_items)):
            item_a, rating_a = rated_items[a]
            for b in range(a + 1, len(rated_items)):
                item_b, rating_b = rated_items[b]
                pair = (item_a, item_b) if item_a < item_b else (item_b, item_a)
                # mantém a ordem do par para somar ri/rj no lado certo
                if item_a < item_b:
                    rating_lo, rating_hi = rating_a, rating_b
                else:
                    rating_lo, rating_hi = rating_b, rating_a
                pair_count[pair] += 1
                sum_product[pair] += rating_lo * rating_hi
                sum_i[pair] += rating_lo
                sum_j[pair] += rating_hi
                sum_sq_i[pair] += rating_lo * rating_lo
                sum_sq_j[pair] += rating_hi * rating_hi
    return pair_count, sum_product, sum_i, sum_j, sum_sq_i, sum_sq_j


#   Pearson sobre o suporte comum + shrinkage:
#   ρ_ij = cov / (σ_i σ_j) (médias tomadas sobre U_ij)
#   s_ij = n_ij / (n_ij + λ2) · ρ_ij
def shrunk_similarities(stats, shrinkage=SIM_SHRINKAGE):
    pair_count, sum_product, sum_i, sum_j, sum_sq_i, sum_sq_j = stats
    similarities = {}
    for pair, num_common in pair_count.items():
        if num_common < 2:
            continue
        cov = sum_product[pair] - sum_i[pair] * sum_j[pair] / num_common
        var_i = sum_sq_i[pair] - sum_i[pair] * sum_i[pair] / num_common
        var_j = sum_sq_j[pair] - sum_j[pair] * sum_j[pair] / num_common
        denom = np.sqrt(var_i * var_j)
        if denom <= 0:
            continue
        correlation = cov / denom
        similarities[pair] = (num_common / (num_common + shrinkage)) * correlation
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


#   Baseline congelada b_ui = μ + b_u + b_i, estimada em forma fechada (paper §2).
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
similarities = shrunk_similarities(cooccurrence_stats(ratings_by_user))
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
