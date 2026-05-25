import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

ratings = pd.read_csv("dataset/ratings.csv")
ratings.head()

# Criar um df(set) semelhanca[user, movie, semelhanca]
# Pij Pearson correlation coefficient entre itens

matriz_ui = ratings.pivot_table(index='userId', columns='movieId', values='rating')
matriz_ui.head()
sim_matriz = matriz_ui.corr(method='pearson', min_periods=2)
sim_matriz.head()
# Nij = numero de user que deram ratings para ambos os itens
# lambda2 = 100
# semelhanca = (Nij  / Nij * lambda2)Pij
# Rui = Bui +
#
