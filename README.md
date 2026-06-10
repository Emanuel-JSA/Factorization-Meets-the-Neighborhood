# Factorization Meets the Neighborhood

Implementação do zero do artigo de Yehuda Koren, _Factorization meets the neighborhood: a multifaceted collaborative filtering model_ (KDD 2008)

📄 Paper: https://doi.org/10.1145/1401890.1401944

Cada modelo é construído sobre o anterior. Todos compartilham **a mesma divisão treino/teste** (`data.py`, seed 42, 20% de teste), então os erros são diretamente comparáveis. Dataset: MovieLens (~100k avaliações).

## Resultados

| Modelo                                         | Arquivo                | RMSE   | MAE    |
| ---------------------------------------------- | ---------------------- | ------ | ------ |
| Baseline `μ + b_u + b_i` (forma fechada)       | `baseline.py`          | 0.9174 | 0.6982 |
| Baseline treinada por gradiente                | `baseline_gd.py`       | 0.8765 | 0.6692 |
| Fatores latentes (SVD)                         | `latent_factor.py`     | 0.8732 | 0.6664 |
| Fatores latentes (SGD)                         | `latent_factor_sgd.py` | 0.8822 | 0.6743 |
| Vizinhança item-item (Pearson)                 | `neighborhood.py`      | 0.8668 | 0.6648 |
| Vizinhança refinada                            | `new_neighborhood.py`  | 0.8474 | 0.6441 |
| **Modelo final — SVD++ + vizinhança (eq. 16)** | `final_model.py`       | 0.8658 | 0.6601 |
