# Implementar un Autoencoder b´asico para 
# las im´agenes binarias de la lista de
# caracteres del archivo ”font.h”
#
# 1. Plantear una arquitectura de red para el Codificador 
# y Decodificar que permita
# representar los datos de entrada en dos dimensiones.
#
# 2. Describan y estudien las diferentes tecnicas de 
# optimizacion que fueron aplicando
# para permitir que la red aprenda todo el set de datos 
# o un subconjunto del
# mismo. En el caso de que sea un subconjunto mostrar 
# porque no fue posible
# aprender el dataset completo.
#
# 3. Realizar el grafico en dos dimensiones que muestre 
# los datos de entrada en el
# espacio latente.
#
# 4. Mostrar como la red puede generar una nueva letra que 
# no pertenece al conjunto
# de entrenamiento.

import matplotlib.pyplot as plt
import numpy as np
from TP5.Ejercicio1.fonts import print_font
from utils.fonts import *
from utils.activations import *
from perceptrons.MultilayerPerceptron import *

character_template = np.array([[0] * 5] * 7)
character = np.copy(font_1[1])
character.resize(7, 1)

bin_array = np.zeros(5, dtype=int)
for i in range(0,5):
    bin_array[4-i] = character[0] & 1
    character[0] >>= 1
bin_array.resize(1, 5)

x_input = font_1
input_names = font_1_symbols

# normalizo datos
x_mean = np.mean(x_input, axis=0)
x_std = np.std(x_input, axis=0)

layers = [
    NeuronLayer(30, 35, activation="tanh"), #35 de entrada
    NeuronLayer(20, activation="tanh"),
    NeuronLayer(2, activation="tanh"), #latent code Z
    NeuronLayer(20, activation="tanh"),
    NeuronLayer(30, activation="tanh"),
    NeuronLayer(35, activation="tanh")
]

def to_binary(x):
    to_ret = []
    for i in x:
        aux = []
        for num in i:
            a = format(num, "b").zfill(5)
            for j in a:
                if j == "0":
                    aux.append(-1)
                elif j == "1":
                    aux.append(1)
        to_ret.append(aux)
    return np.array(to_ret)

def to_decimal(x):
    x = (x+1)/2

    to_detransform = np.rint(x*255)
    for i in range(len(to_detransform)):
        if to_detransform[i] > 255:
            to_detransform[i] = 255
        if to_detransform[i] < 0:
            to_detransform[i] = 0
    return to_detransform

x = to_binary(x_input)

autoencoder = MultiLayerPerceptron(layers, eta=0.0001)

min_error, errors, ii, training_accuracies, test_accuracies, min_error_test = autoencoder.train(
    x, x, 1, iterations_qty=30000)

print('MIN ERROR: ', min_error)

encoder = MultiLayerPerceptron(autoencoder.neuron_layers[0:int(len(layers)/2)])
decoder = MultiLayerPerceptron(autoencoder.neuron_layers[int(len(layers)/2):])

aux_1 = []
aux_2 = []
for i in range(len(x)):
    to_predict = x[i, :]
    encoded = encoder.predict(to_predict)
    decoded = decoder.predict(encoded)
    print(f"{(to_predict)} -> {encoded} -> {to_decimal(decoded)}" )
    print(f"{to_predict} -> {decoded}")
    print_font(to_predict)#.astype(np.int64))
    print()
    print()
    print_font(decoded)#.astype(np.int64))
    aux_1.append(encoded[0])
    aux_2.append(encoded[1])
    print("(",decoded[0],",",decoded[1],")")

print(aux_1)
print(aux_2)

#ej 1.a 4)
print("\n\n\n-------------------------\n")
values = [-1,0,1]
for i in values:
    for j in values:
        print(i, " ", j, ": ")
        new_letter = [i, j]
        decoded = decoder.predict(new_letter)
        print_font(decoded)

plt.xlim([-1.1, 1.1])
plt.ylim([-1.1, 1.1])
for i, txt in enumerate(input_names):
    plt.annotate(txt, (aux_1[i], aux_2[i]))
plt.scatter(aux_1, aux_2)
plt.show()
