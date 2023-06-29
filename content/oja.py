import numpy as np
from get_data import data, raw_data
from Neuron import Neuron
from utils import * 

def update_weigth(eta, s, weights, j):
    return weights + eta * s * (data[j] - s * weights)

def activation(weights, j):
    s = 0
    for i in range(len(data[j])):
        s += data[j][i] * weights[i]
    return s

def oja(epochs = 5000, eta = 0.0001):
    print("DATA", data)
    N = len(data)
    weight = np.random.uniform(-1, 1, len(data[0]))
    w = []
    w.append(weight)
    for i in range(0, epochs):
        for j in range(0, N):
            s = activation(weight, j)
            weight = update_weigth(eta, s, weight, j)
            w.append(weight)
    print("Autovector: ", w[-1])

    f_c_o = np.matmul(data, w[-1])

    print("Primera componente: ", f_c_o)


oja()