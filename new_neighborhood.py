import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# R^k(i,u) => K vizinhos de i que o u avaliou
# N^k(i,u) =: K vizinhos de i que u interagiu implicitamente
# rui = bui + |R^k(i,u)|^-1/2 * jeR^k(i,u) (ruj - buj)wij + |N^k(i,u)|^-1/2 * jeN^k(i,u) cij

# Pegar vizinhos