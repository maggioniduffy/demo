from cProfile import label
import csv
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from sys import stdout
from math import exp, floor
import parser
from network import Network
from constants import ModeOptions
from utils import createNoise, predictAndPrintResults, concatenateArrays
from graphing import plotLatentSpace

CONFIG_INPUT = "configuration.json"
labels = ['@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_']

def trainMultilayer(config, inputs):
    network = Network(config, inputs.shape[1])
    latent, latentLabels = network.train(inputs, inputs, labels)
    if len(config.generatorPoints) > 0:
        results = network.generate(config.generatorPoints)
        results = [[r[0], np.array([1 if e > 0.5 else 0 for e in r[1]]).reshape((7, 5))] for r in results]
        for result in results:
            print(f'Generated using {result[0]}:\n {result[1]}')
        plotLatentSpace(latent,latentLabels,generated = results)
    return network

def trainMultilayerOptimizer(config, inputs, optimizer):
    network = Network(config, inputs.shape[1])
    error = network.trainMinimizer(inputs, optimizer)

def trainDenoiser(config, inputs):
    repeatedInput = 3
    # # Which inputs to use to generate noise
    indexesSample = [7, 14, 16, 17, 21, 24, 25, 26, 29, 30]
    # # inputsCount = len(inputs)
    # # indexes = [ x for x in range(0, inputsCount) ]
    # # indexesSample = random.sample(indexes, config.noiseCount if config.noiseCount < inputsCount else inputsCount)
    
    inputCharacters = []
    expected = []
    for index in indexesSample:
        inputCharacters.append(inputs[index])
        for _ in range(0, repeatedInput):
             expected.append(inputs[index])

    # Expected outcome 
    expected = np.copy(np.array(inputs))
    # Create noise with expected input
    noiseInput = np.array([createNoise(origInput,2/35) for origInput in expected])
    
    # Create instance of the network
    network = Network(config, inputs.shape[1])
    # Train with noise
    network.train(noiseInput, expected, labels)

    print('#### TRAINING SET RESULTS ####')

    predictAndPrintResults(network, noiseInput, expected, plot = False)
    print('---')
    predictNewNoise(config, network, inputs)
    
def predictNewNoise(config, network, expected):
    print('#### NEW SET RESULTS ####')

    noiseInput = np.array([createNoise(origInput,2/35) for origInput in expected])
    
    predictAndPrintResults(network, noiseInput, expected)

def trainGenerative(config):
    # Parse input images
    inputs = parser.parseInput(config.input)
    labels = ['@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_']
    # Create instance of the network
    network = Network(config, inputs[0].shape[0])
    # Train the network
    network.train(inputs, inputs, labels)
    n = 25
    dimensions = (7, 5)
    grid_x = np.linspace(0.05, 0.95, n)
    grid_y = np.linspace(0.05, 0.95, n)
    figure = np.zeros((dimensions[0] * n, dimensions[1] * n))
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            # Include bias in sample
            z_sample = np.array([1, xi, yi])
            print('z sample: ', z_sample)
            x_decoded = network.generateFromPoint(z_sample)
            x_decoded = np.array([0 if e > 0.5 else 1 for e in x_decoded])
            digit = x_decoded.reshape(dimensions[0], dimensions[1])
            figure[i * dimensions[0]: (i + 1) * dimensions[0],
                j * dimensions[1]: (j + 1) * dimensions[1]] = digit
    plt.figure(figsize=(7, 7))
    plt.imshow(figure, cmap='Greys_r')
    plt.show()
    
def main():
    config = parser.parseConfiguration(CONFIG_INPUT)    

    print("######################\nTRAINING\n######################")
    if config.mode == ModeOptions.NORMAL.value:
        inputs = parser.parseInput(config.input)
        trainMultilayer(config, inputs)
    elif config.mode == ModeOptions.DENOISER.value:
        inputs = parser.parseInput(config.input)
        trainDenoiser(config, inputs)
    elif config.mode == ModeOptions.GENERATIVE.value:
        inputs = parser.parseInput(config.input)
        trainGenerative(config)
    else:
        inputs = parser.parseInput(config.input)
        trainMultilayerOptimizer(config, inputs, config.optimizer)

if __name__ == "__main__":
    main()