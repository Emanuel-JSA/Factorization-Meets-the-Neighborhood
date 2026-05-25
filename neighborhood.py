import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

ratings = pd.read_csv("dataset/ratings.csv")
ratings.head()

# Pij Pearson correlation coefficient entre itens
matriz_ui = ratings.pivot_table(index='userId', columns='movieId', values='rating')
matriz_ui.head()
pearson_matriz = matriz_ui.corr(method='pearson', min_periods=2)
pearson_matriz.head()

# Nij número de usuários que avaliaram ambos os itens
mask = matriz_ui.notna()
n_ij = mask.T.dot(mask)
n_ij.head()

# Semelhança entre os itens
lambda2 = 100
s_matriz = n_ij / (n_ij + lambda2) * pearson_matriz
s_matriz.head()
