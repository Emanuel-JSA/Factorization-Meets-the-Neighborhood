import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


# Criar um df(set) semelhanca[user, movie, semelhanca]
# Pij Pearson correlation coefficient
# Nij = numero de user que deram ratings para ambos os itens
# lambda2 = 100
# semelhanca = (Nij  / Nij * lambda2)Pij

# Rui = Bui +
