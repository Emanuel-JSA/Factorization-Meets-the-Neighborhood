import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data import test_df as test
from data import train_df as train

# Hiperparâmetros
alpha = 0.005
lambda1 = 0.01
epocas = 50

# descobrir a media geral
mu = train["rating"].mean()

# Inicializa todos os bias em 0
usuarios = train["userId"].unique()
filmes = train["movieId"].unique()

b_u = {u: 0.0 for u in usuarios}
b_i = {i: 0.0 for i in filmes}

historico_rmse = []

for epoca in range(epocas):
    train_sample = train.sample(frac=1, random_state=epoca)

    for row in train_sample.itertuples():
        u = row.userId
        i = row.movieId
        r = row.rating

        e_ui = r - mu - b_u.get(u, 0.0) - b_i.get(i, 0.0)

        b_u[u] = b_u.get(u, 0.0) + alpha * (e_ui - lambda1 * b_u.get(u, 0.0))
        b_i[i] = b_i.get(i, 0.0) + alpha * (e_ui - lambda1 * b_i.get(i, 0.0))

    preds = [
        mu + b_u.get(row.userId, 0.0) + b_i.get(row.movieId, 0.0)
        for row in test.itertuples()
    ]
    rmse = np.sqrt(mean_squared_error(test["rating"], preds))
    historico_rmse.append(rmse)

# print(
#     f"\nMelhor RMSE: {min(historico_rmse):.4f} (época {historico_rmse.index(min(historico_rmse)) + 1})"
# )

test = test.copy()
test["predicao"] = [
    mu + b_u.get(row.userId, 0.0) + b_i.get(row.movieId, 0.0)
    for row in test.itertuples()
]

# df_rmse = pd.DataFrame({"epoca": range(1, epocas + 1), "rmse": historico_rmse})

# df_rmse.plot(x="epoca", y="rmse", marker="o", figsize=(10, 5), legend=False)
# plt.xlabel("Época")
# plt.ylabel("RMSE")
# plt.title("Convergência do modelo baseline com SGD")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

rmse = np.sqrt(mean_squared_error(test["rating"], test["predicao"]))
mae = mean_absolute_error(test["rating"], test["predicao"])

print(f"RMSE= {rmse}")
print(f"MAE= {mae}")
# RMSE= 0.8667137117752014
# MAE= 0.6627077073075438

# %%
# from itertools import product

# alphas = [0.001, 0.005, 0.01]
# lambdas = [0.01, 0.02, 0.05]
# epocas = 40

# resultados = []

# for alpha, lambda1 in product(alphas, lambdas):
#     b_u = {u: 0.0 for u in train["userId"].unique()}
#     b_i = {i: 0.0 for i in train["movieId"].unique()}

#     for epoca in range(epocas):
#         train_shuffled = train.sample(frac=1, random_state=epoca)
#         for row in train_shuffled.itertuples():
#             u, i, r = row.userId, row.movieId, row.rating
#             e_ui = r - mu - b_u.get(u, 0.0) - b_i.get(i, 0.0)
#             b_u[u] = b_u.get(u, 0.0) + alpha * (e_ui - lambda1 * b_u.get(u, 0.0))
#             b_i[i] = b_i.get(i, 0.0) + alpha * (e_ui - lambda1 * b_i.get(i, 0.0))

#     preds = [mu + b_u.get(r.userId, 0.0) + b_i.get(r.movieId, 0.0) for r in test.itertuples()]
#     rmse = np.sqrt(mean_squared_error(test["rating"], preds))

#     resultados.append({"alpha": alpha, "lambda1": lambda1, "rmse": rmse})
#     print(f"alpha={alpha} lambda={lambda1} | RMSE={rmse:.4f}")

# df_resultados = pd.DataFrame(resultados)
# print("\nMelhor combinação:")
# print(df_resultados.loc[df_resultados["rmse"].idxmin()])
