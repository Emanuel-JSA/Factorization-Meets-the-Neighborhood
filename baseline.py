import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data import test_df as test
from data import train_df as train

# descobrir a media geral
media_geral = train["rating"].mean()
# descobrir o desvio da media do usuario
media_usuarios = train.groupby("userId")["rating"].mean()
desvio_usuarios = media_usuarios - media_geral

# descobrir o desvio da media do filme
media_filmes = train.groupby("movieId")["rating"].mean()
desvio_filmes = media_filmes - media_geral

test = test.copy()
test["b_u"] = test["userId"].map(desvio_usuarios).fillna(0)
test["b_i"] = test["movieId"].map(desvio_filmes).fillna(0)
test["predicao"] = media_geral + test["b_u"] + test["b_i"]

rmse = np.sqrt(mean_squared_error(test["rating"], test["predicao"]))
mae = mean_absolute_error(test["rating"], test["predicao"])

print(f"RMSE= {rmse}")
print(f"MAE= {mae}")
# RMSE= 0.9115849654884866
# MAE= 0.6944658365734805
