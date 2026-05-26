import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import product
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# =====================
# CONFIG DO EXPERIMENTO
# =====================
GRID_K          = [10, 20, 50]          # tamanho da vizinhanca
GRID_LMBDA_THETA = [0.02, 0.05, 0.1]    # regularizacao do theta
SEEDS           = [10, 20, 30]          # seeds de split (media +- desvio)

# Hiperparametros fixos
LMBDA_B   = 0.02       # regularizacao dos baselines
ALPHA0    = 0.01       # learning rate inicial
DECAY     = 0.9        # multiplicador do lr por epoca
EPOCAS    = 40
PACIENCIA = 3          # early stopping
LAMBDA2   = 100        # shrinkage da similaridade
MIN_PER   = 5          # min_periods do Pearson

ratings = pd.read_csv("dataset/ratings.csv")


def construir_similaridade(train):
    """Calcula s_ij = n_ij/(n_ij+lambda2) * pearson_ij a partir do TREINO."""
    matriz_ui = train.pivot_table(index="userId", columns="movieId", values="rating")
    pearson = matriz_ui.corr(method="pearson", min_periods=MIN_PER)
    mask = matriz_ui.notna().astype(int)
    n_ij = mask.T @ mask
    s = n_ij / (n_ij + LAMBDA2) * pearson
    return s


def construir_vizinhanca(s_matriz, k):
    """Top-k vizinhos de cada item segundo s_ij. Recorta da matriz ja pronta."""
    viz = {}
    for i in s_matriz.columns:
        sims = s_matriz[i].drop(i, errors="ignore").dropna()
        viz[i] = list(sims.sort_values(ascending=False).head(k).index)
    return viz


def treinar_e_avaliar(train, test, vizinhanca, lmbda_theta):
    """Treina o modelo de vizinhanca com theta aprendido. Retorna (rmse, mae)."""
    ratings_usuario = defaultdict(dict)
    for row in train.itertuples():
        ratings_usuario[row.userId][row.movieId] = row.rating

    mu = train["rating"].mean()
    b_u = defaultdict(float)
    b_i = defaultdict(float)
    theta = defaultdict(float)

    def baseline(u, i):
        return mu + b_u[u] + b_i[i]

    def predizer(u, i):
        pred = baseline(u, i)
        avaliados = ratings_usuario.get(u, {})
        for j in vizinhanca.get(i, []):
            if j in avaliados:
                pred += theta[(i, j)] * (avaliados[j] - baseline(u, j))
        return pred

    melhor_rmse = float("inf")
    melhor_mae = float("inf")
    melhor_estado = None
    contador = 0
    alpha = ALPHA0

    for epoca in range(EPOCAS):
        train_sample = train.sample(frac=1, random_state=epoca)

        for row in train_sample.itertuples():
            u, i, r = row.userId, row.movieId, row.rating
            avaliados = ratings_usuario.get(u, {})

            pred = baseline(u, i)
            vizinhos_validos = []
            for j in vizinhanca.get(i, []):
                if j in avaliados and j != i:
                    desvio = avaliados[j] - baseline(u, j)
                    pred += theta[(i, j)] * desvio
                    vizinhos_validos.append((j, desvio))

            e_ui = r - pred
            b_u[u] += alpha * (e_ui - LMBDA_B * b_u[u])
            b_i[i] += alpha * (e_ui - LMBDA_B * b_i[i])
            for j, desvio in vizinhos_validos:
                theta[(i, j)] += alpha * (e_ui * desvio - lmbda_theta * theta[(i, j)])

        preds = [predizer(row.userId, row.movieId) for row in test.itertuples()]
        rmse = np.sqrt(mean_squared_error(test["rating"], preds))
        mae = mean_absolute_error(test["rating"], preds)

        if rmse < melhor_rmse:
            melhor_rmse, melhor_mae = rmse, mae
            melhor_estado = (dict(b_u), dict(b_i), dict(theta))
            contador = 0
        else:
            contador += 1
            if contador >= PACIENCIA:
                break

        alpha *= DECAY

    return melhor_rmse, melhor_mae


def baseline_puro(train, test):
    """RMSE/MAE so com mu + b_u + b_i. Piso de comparacao."""
    mu = train["rating"].mean()
    b_u = defaultdict(float)
    b_i = defaultdict(float)
    alpha = ALPHA0
    for epoca in range(EPOCAS):
        for row in train.sample(frac=1, random_state=epoca).itertuples():
            e = row.rating - (mu + b_u[row.userId] + b_i[row.movieId])
            b_u[row.userId] += alpha * (e - LMBDA_B * b_u[row.userId])
            b_i[row.movieId] += alpha * (e - LMBDA_B * b_i[row.movieId])
        alpha *= DECAY
    preds = [mu + b_u[r.userId] + b_i[r.movieId] for r in test.itertuples()]
    return (np.sqrt(mean_squared_error(test["rating"], preds)),
            mean_absolute_error(test["rating"], preds))


# ======================================================================
# LOOP PRINCIPAL DO EXPERIMENTO
# ======================================================================
resultados = []

for seed in SEEDS:
    train, test = train_test_split(ratings, test_size=0.2, random_state=seed)

    # baseline puro (piso)
    rmse_b, mae_b = baseline_puro(train, test)
    resultados.append({"modelo": "baseline", "K": None, "lmbda_theta": None,
                       "seed": seed, "rmse": rmse_b, "mae": mae_b})
    print(f"[seed {seed}] baseline           RMSE={rmse_b:.4f}  MAE={mae_b:.4f}")

    # similaridade calculada UMA vez por seed; vizinhanca recortada por K
    s_matriz = construir_similaridade(train)

    for k in GRID_K:
        vizinhanca = construir_vizinhanca(s_matriz, k)
        for lt in GRID_LMBDA_THETA:
            rmse, mae = treinar_e_avaliar(train, test, vizinhanca, lt)
            resultados.append({"modelo": "theta", "K": k, "lmbda_theta": lt,
                               "seed": seed, "rmse": rmse, "mae": mae})
            print(f"[seed {seed}] K={k:<3} lmbda_theta={lt:<5} "
                  f"RMSE={rmse:.4f}  MAE={mae:.4f}")


df = pd.DataFrame(resultados)

resumo = (df.groupby(["modelo", "K", "lmbda_theta"], dropna=False)
            .agg(rmse_media=("rmse", "mean"),
                 rmse_std=("rmse", "std"),
                 mae_media=("mae", "mean"),
                 mae_std=("mae", "std"))
            .reset_index()
            .sort_values("rmse_media"))

resumo.head()
