import numpy as np
from scipy.optimize import minimize
from perceptron import Perceptron
from utils import printIterationsInPlace, getRandomDatasetOrder
from graphing import plotLatentSpace, plotError

class Network:

    def __init__(self, config, inputSize):
        self.inputSize = inputSize
        self.network = self.new_network(config, inputSize)
        self.networkSize = self.network.shape[0]
        self.config = config
        self.latentCode = []
        self.optimizerError = []

    def new_network(self, config, inputSize):
        network = []
        isFirst = True
        lastLayer = []
        for layer in config.layers:
            if isFirst:
                weightCount = inputSize
                isFirst = False
            else:
                weightCount = lastLayer[1] + 1
            network.append(np.array([Perceptron(weightCount, layer[0], config.learningRate, config.beta,config.momentum, config.alpha) for x in range(0, layer[1])], dtype=Perceptron))
            lastLayer = layer
        return np.array(network, dtype=object)

    def flat_network(self):
        weightMatrix = []
        for layer in self.network:
            for perceptron in layer:
                weightMatrix.append(perceptron.weights_get()) 
        
        weightMatrix = np.array(weightMatrix, dtype=object)
        return np.hstack(weightMatrix)

    def unflat_weights(self, flatWeights):
        network = []
        isFirst = True
        lastLayer = []
        currIndex = 0
        for layer in self.config.layers:
            if isFirst:
                weightCount = self.inputSize
                isFirst = False
            else:
                weightCount = lastLayer[1] + 1

            layerWeights = []
            for _ in range(0, layer[1]):
                layerWeights.append(flatWeights[currIndex:currIndex+weightCount])
                currIndex += weightCount
            
            network.append(np.array(layerWeights))
            lastLayer = layer
        return np.array(network, dtype=object)

    def network_builder(self, flatWeights):
        weightMatrix = self.unflat_weights(flatWeights)

        for row in range(0, weightMatrix.shape[0]):
            for col in range(0, self.config.layers[row][1]):
                self.network[row][col].weights_set(weightMatrix[row][col])
        
    def forward_propagation(self, input, err=False, storeLatent = True, offsetStart = 0, offsetValues = [0,0]):
        activationValues = []
        summationValues = []
        for index, layer in enumerate(self.network):
            if index < offsetStart - 1:
                activationValues.append(np.array([]))
                summationValues.append(np.array([]))
            elif index == offsetStart - 1:
                activationValues.append(np.array([1] + offsetValues))
                summationValues.append(np.array(offsetValues))
            else:
                data = input if index == 0 else activationValues[index - 1]
                summationValues.append(
                    np.array([perceptron.sum(data) for perceptron in layer]))
                activations = [layer[i].activate(summationValues[index][i]) for i in range(len(summationValues[index]))]
                if not err and storeLatent and layer.shape[0] == 2:
                    self.latentCode.append(activations)
                if index < self.networkSize - 1:
                    activations = [1] + activations
                activationValues.append(np.array(activations))
        return summationValues, activationValues

    def back_propagation(self, input, summations, activations):
        initialBackpropagation = [perceptron.backpropagate_init(
            summations[-1][index], input[index + 1], activations[-1][index]) for index, perceptron in enumerate(self.network[-1])]
        backpropagationValues = [None] * self.networkSize
        backpropagationValues[-1] = np.array(initialBackpropagation)
        for index in range(self.networkSize - 2, -1, -1):
            data = []
            for subindex, perceptron in enumerate(self.network[index]):
                outboundWeights = np.array([p.weights[subindex + 1] for p in self.network[index + 1]])
                data.append(perceptron.backpropagate(summations[index][subindex], outboundWeights, backpropagationValues[index + 1]))
            backpropagationValues[index] = np.array(data)
        return backpropagationValues

    def weights(self, input, backpropagations, activations):
        for index, layer in enumerate(self.network):
            data = input if index == 0 else activations[index - 1]
            for subindex, p in enumerate(layer):
                p.update_weights(backpropagations[index][subindex], data)

    def input_error(self, actualData, predictedData):
        return np.linalg.norm(actualData-predictedData)**2

    def error_calculation(self, input, expected):
        error = 0
        for i in range(len(input)):
            _, activ = self.forward_propagation(input[i], err=True)
            error = error + self.input_error(expected[i][1:], activ[-1])
        return error

    def predict(self, input):
        _, activ = self.forward_propagation(input)
        return activ[-1]

    def generate(self, latentInputs):
        startIndex = np.ceil(self.networkSize/2)
        results = []
        for latentInput in latentInputs:
            _, activ = self.forward_propagation(input, offsetStart=startIndex, offsetValues=latentInput)
            results.append([latentInput, activ[-1]])
        return results

    def generateFromPoint(self, latentInput):
        startIndex = np.ceil(self.networkSize/2)
        _, activ = self.forward_propagation(input, offsetStart=startIndex, offsetValues=latentInput)
        return activ[-1]

    def train(self, input, expected, labels):
        trainingSize = input.shape[0]
        iterations = 0
        error = 1
        errors = []
        latentLabels = []
        try:
            while iterations < self.config.iterations and error > self.config.error:
                printIterationsInPlace(iterations)
                indexes = getRandomDatasetOrder(trainingSize)
                self.latentCode = []
                latentLabels = []
                for itemIndex in indexes:
                    latentLabels.append(labels[itemIndex])
                    summationValues, activationValues = self.forward_propagation(input[itemIndex])
                    backpropagationValues = self.back_propagation(input[itemIndex], summationValues, activationValues)
                    self.weights(input[itemIndex], backpropagationValues, activationValues)
                error = self.error_calculation(input, expected)/trainingSize
                errors.append(error)
                iterations += 1
            print(f'Final loss is {errors[-1]}')
            plotError(errors)
            plotLatentSpace(self.latentCode, latentLabels)
            return self.latentCode, latentLabels
        except KeyboardInterrupt:
            print("Finishing up...")


    def cost(self, flatWeights, input, expected):
        self.network_builder(flatWeights)
        error = self.error_calculation(input, expected)/input.shape[0]
        self.optimizerError.append(error)
        print('ERROR', error)
        return error

    def trainMinimizer(self, input, optimizer):
        flattenedWeights = self.flat_network()
        res = minimize(fun=self.cost, x0=flattenedWeights, args=(input, input), method=optimizer, options={'maxfun': self.config.iterations, 'maxfev': self.config.iterations, 'maxiter': 1, 'disp': True})
        self.network_builder(flattenedWeights)
        error = res.fun
        print(f'Final loss is {error}')
        plotError(self.optimizerError)
        return error