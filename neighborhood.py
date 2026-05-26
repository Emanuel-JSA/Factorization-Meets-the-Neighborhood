import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

ratings = pd.read_csv("dataset/ratings.csv")
train, test = train_test_split(ratings, test_size=0.2, random_state=10)

# SIMILARIDADE s_ij  (usada SO para escolher vizinhos, nao para o peso)
#    s_ij = n_ij / (n_ij + lambda2) * pearson_ij
matriz_ui = train.pivot_table(index="userId", columns="movieId", values="rating")

pearson_matriz = matriz_ui.corr(method="pearson", min_periods=2)

# n_ij = numero de usuarios que avaliaram AMBOS os itens.
# Truque: mascara binaria (1 onde tem rating) e produto matricial.
mask = matriz_ui.notna().astype(int)
n_ij = mask.T @ mask

lambda2 = 100
s_matriz = n_ij / (n_ij + lambda2) * pearson_matriz


def vizinhos_do_item(movie_j, k=20):
    if movie_j not in s_matriz.columns:
        return []
    sims = s_matriz[movie_j].drop(movie_j, errors="ignore").dropna()
    return list(sims.sort_values(ascending=False).head(k).index)


# Pre-computa a vizinhanca de cada item UMA vez.
K = 10
vizinhanca = {i: vizinhos_do_item(i, K) for i in s_matriz.columns}

# buscar r_uj em O(1) dentro do loop de treino.
ratings_usuario = defaultdict(dict)
for row in train.itertuples():
    ratings_usuario[row.userId][row.movieId] = row.rating

alpha = 0.005
lmbda = 0.01
epocas = 4

mu = train["rating"].mean()
b_u = defaultdict(float)
b_i = defaultdict(float)
# theta esparso: chave (i, j). Criado sob demanda, default 0.0
theta = defaultdict(float)


def baseline(u, i):
    return mu + b_u[u] + b_i[i]


def predizer(u, i):
    """r_hat_ui = b_ui + sum_{j in S^k(i;u)} theta_ij (r_uj - b_uj)."""
    pred = baseline(u, i)
    avaliados_por_u = ratings_usuario.get(u, {})
    for j in vizinhanca.get(i, []):
        if j in avaliados_por_u: # interseccao = S^k(i;u)
            desvio = avaliados_por_u[j] - baseline(u, j)
            pred += theta[(i, j)] * desvio
    return pred


# TREINO (SGD)
historico_rmse = []

for epoca in range(epocas):
    train_sample = train.sample(frac=1, random_state=epoca)

    for row in train_sample.itertuples():
        u, i, r = row.userId, row.movieId, row.rating

        avaliados_por_u = ratings_usuario.get(u, {})

        # Predicao com os parametros atuais
        pred = baseline(u, i)
        vizinhos_validos = []
        for j in vizinhanca.get(i, []):
            if j in avaliados_por_u and j != i:
                desvio = avaliados_por_u[j] - baseline(u, j)
                pred += theta[(i, j)] * desvio
                vizinhos_validos.append((j, desvio))

        e_ui = r - pred

        # Updates (mesmas regras derivadas analiticamente)
        b_u[u] += alpha * (e_ui - lmbda * b_u[u])
        b_i[i] += alpha * (e_ui - lmbda * b_i[i])
        for j, desvio in vizinhos_validos:
            theta[(i, j)] += alpha * (e_ui * desvio - lmbda * theta[(i, j)])

    # Avaliacao no teste
    preds = [predizer(row.userId, row.movieId) for row in test.itertuples()]
    rmse = np.sqrt(mean_squared_error(test["rating"], preds))
    historico_rmse.append(rmse)
    print(f"epoca {epoca + 1:2d}  RMSE_teste = {rmse:.4f}")

print("\nMelhor RMSE:", min(historico_rmse))

test = test.copy()
test["predicao"] = [
    predizer(row.userId, row.movieId)
    for row in test.itertuples()
]

rmse = np.sqrt(mean_squared_error(test["rating"], test["predicao"]))
mae = mean_absolute_error(test["rating"], test["predicao"])

print(f"RMSE= {rmse}")
print(f"MAE= {mae}")

# RMSE= 0.8668186936661175
# MAE= 0.6648198293001096
