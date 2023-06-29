from math import tanh, copysign
import numpy as np
from constants import ActivationOptions 

class Perceptron:
    def __init__(self, weightsAmount, activationMethod, learningRate, beta, momentum, alpha):
        
        self.beta = beta
        self.alpha = alpha
        self.activationMethod = activationMethod
        self.weights = np.random.rand(weightsAmount) * np.sqrt(1/weightsAmount)
        self.weightsAmount = weightsAmount
        self.activation = self.activation_function(activationMethod)
        self.derivative = self.activation_d_funcion(activationMethod)
        self.learningRate = learningRate
        self.previousCorrection = np.zeros(weightsAmount)
        self.useMomentum = momentum

    def weights_get(self):
        return np.copy(self.weights)

    def weights_set(self, weights):
        self.weights = weights

    def sum(self, inputs):
        return np.dot(inputs, self.weights) 
    
    def activate(self, sum):
        return self.activation(sum)

    def backpropagate(self, sum, otherWeights, backpropagations):
        return self.derivative(sum) * np.dot(otherWeights, backpropagations) 

    def backpropagate_init(self, sum, desired, prediction):
        return self.derivative(sum) * (desired - prediction)

    def initialBackpropagateWithError(self, sum, error):
        return self.derivative(sum) * error

    def update_weights(self, backpropagation, prediction):
        correction = self.learningRate * backpropagation * prediction
        if self.useMomentum:
            correction = correction + self.alpha * self.previousCorrection
            self.previousCorrection = correction

        self.weights = self.weights + correction

    def calculateError(self, desired, prediction):
        return ((desired - prediction)**2) * 0.5

    def simple(self, sum):
        return copysign(1, sum)

    def simple_d(self, sum):
        return 1

    def linear(self, sum):
        return sum

    def linear_d(self, sum):
        return 1

    def nonLinear(self, sum):
        return tanh(self.beta * sum)

    def nonlinear_d(self, sum):
        return self.beta * (1 - (self.nonLinear(sum)**2))

    def relu(self, sum):
        return max(0, sum)

    def relu_d(self, sum):
        return 0 if sum <= 0 else 1

    def __str__(self):
        subs = 'activation=%s, weights=%s, learningRate=%s' % (self.activationMethod, self.weights, self.learningRate)
        s = '%s{%s}' % (type(self).__name__, subs)
        return s

    def activation_function(self, activationMethod):
        activations = {
            ActivationOptions.SIMPLE.value: self.simple, 
            ActivationOptions.LINEAR.value: self.linear, 
            ActivationOptions.NON_LINEAR.value: self.nonLinear,
            ActivationOptions.RELU.value: self.relu
        }
        return activations[activationMethod]

    def activation_d_funcion(self, activationMethod):
        derivatives = {
            ActivationOptions.SIMPLE.value: self.simple_d, 
            ActivationOptions.LINEAR.value: self.linear_d, 
            ActivationOptions.NON_LINEAR.value: self.nonlinear_d,
            ActivationOptions.RELU.value: self.relu_d
        }
        return derivatives[activationMethod]

    

    
