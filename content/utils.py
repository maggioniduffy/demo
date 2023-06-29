from re import A
import numpy as np
from sympy import rad
from get_data import data

def set_init_weights(k):
    w = []
    for _ in range(0,k*k):
      i = (int) (np.random.normal())
      wi = data[i]
      w.append(wi)
    return np.array(w)
  
def update_neighborhood_weight(weights, radius, w_k, k):
  aux = np.empty((k,k),list)
  index = 0
  pos = None
  for i in range(k):
    for j in range(k):
      aux[i][j] = weights[index]
      if index == w_k:
        pos = np.array([i,j])
      index += 1
  res = []
  
  index = 0
  for i in range(k):
    for j in range(k):
      if index != w_k and np.linalg.norm(np.array([i,j]) - pos) <= radius:
        res.append(index)
      index += 1
  return np.array(res)

def get_winner_neuron(x,weights):
  aux = list(map(lambda wi: np.linalg.norm(x-wi), weights))
  aux = np.array(aux)
  w_k = np.argmin(aux)
  return w_k