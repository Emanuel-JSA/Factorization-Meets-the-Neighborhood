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

K = 40
LAMBDA2 = 100

ratings = pd.read_csv("dataset/ratings.csv")


# R(u) = itens que u avaliou, R(i) = usuários que avaliaram i
def build_lookups(df):
    Ru = defaultdict(dict)  # user -> {item: rating}
    Ri = defaultdict(dict)  # item -> {user: rating}
    for u, i, r in zip(df.userId.values, df.movieId.values, df.rating.values):
        Ru[u][i] = r
        Ri[i][u] = r
    return Ru, Ri


#   Estatísticas de co-avaliação, acumuladas varrendo cada usuário uma vez.
#   Para cada par (i, j) avaliado pelo mesmo usuário acumulamos as somas
#   necessárias para a correlação de Pearson sobre o suporte comum U_ij.
#   Custo ~ Σ_u |R(u)|^2  (esparso, evita o item×item denso).
def cooccurrence_stats(Ru):
    n = defaultdict(int)  # |U_ij|  nº de usuários que avaliaram i e j
    sij = defaultdict(float)  # Σ r_ui r_uj
    si = defaultdict(float)  # Σ r_ui
    sj = defaultdict(float)  # Σ r_uj
    sii = defaultdict(float)  # Σ r_ui^2
    sjj = defaultdict(float)  # Σ r_uj^2
    for items in Ru.values():
        keys = list(items.items())
        for a in range(len(keys)):
            i, ri = keys[a]
            for b in range(a + 1, len(keys)):
                j, rj = keys[b]
                key = (i, j) if i < j else (j, i)
                # mantém a ordem do par para somar ri/rj no lado certo
                if i < j:
                    ri_, rj_ = ri, rj
                else:
                    ri_, rj_ = rj, ri
                n[key] += 1
                sij[key] += ri_ * rj_
                si[key] += ri_
                sj[key] += rj_
                sii[key] += ri_ * ri_
                sjj[key] += rj_ * rj_
    return n, sij, si, sj, sii, sjj


#   Pearson sobre o suporte comum + shrinkage:
#   ρ_ij = cov / (σ_i σ_j) (médias tomadas sobre U_ij)
#   s_ij = n_ij / (n_ij + λ2) · ρ_ij
def shrunk_similarities(stats, lam=LAMBDA2):
    n, sij, si, sj, sii, sjj = stats
    sim = {}
    for key, nij in n.items():
        if nij < 2:
            continue
        cov = sij[key] - si[key] * sj[key] / nij
        vi = sii[key] - si[key] * si[key] / nij
        vj = sjj[key] - sj[key] * sj[key] / nij
        denom = np.sqrt(vi * vj)
        if denom <= 0:
            continue
        rho = cov / denom
        sim[key] = (nij / (nij + lam)) * rho
    return sim


#   Top-K vizinhos por item: N(i) = K itens j com maior s_ij.
#   sim é simétrico (guardamos só o par ordenado), então alimenta os dois lados.
def topk_neighbors(sim, k=K):
    by_item = defaultdict(list)  # item -> [(sim, vizinho), ...]
    for (i, j), s in sim.items():
        by_item[i].append((s, j))
        by_item[j].append((s, i))
    N = {}
    for i, lst in by_item.items():
        lst.sort(reverse=True)
        N[i] = [j for _, j in lst[:k]]
    return N


Ru, Ri = build_lookups(ratings)
sim = shrunk_similarities(cooccurrence_stats(Ru))
N = topk_neighbors(sim)


# R^k(i,u) = N(i) ∩ R(u)  — interseção barata feita na hora da predição
def neighbors_rated_by(u, i):
    return [j for j in N.get(i, []) if j in Ru[u]]


print("n pares de similaridade:", len(sim))
i = next(iter(N))
print("exemplo item", i, "->", N[i][:5])
u = next(iter(Ru))
print("R^k para (u=%s, i=%s):" % (u, i), neighbors_rated_by(u, i)[:8])
