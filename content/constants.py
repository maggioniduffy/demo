import enum

FILES = "files"
LAYERS = "layers"

class ActivationOptions(enum.Enum):
   SIMPLE = "simple"
   LINEAR = "linear"
   NON_LINEAR = "nonlinear"
   RELU = "relu"

   def __str__(self):
      return str(self.value)

class ModeOptions(enum.Enum):
   NORMAL = "normal"
   OPTIMIZER = "optimizer"
   DENOISER = "denoiser"
   GENERATIVE = "generative"
   def __str__(self):
      return str(self.value)

class ConfigOptions(enum.Enum):
   INPUT_DATA = "input"
   ITERATIONS = "iterations"
   LEARNING_RATE = "learningRate"
   ERROR_LIMIT = "error"
   BETA = "beta"
   MOMENTUM = "momentum"
   ALPHA = "alpha"
   ACTIVATION = "activation"
   PERCEPTRONS = "perceptrons"
   MODE = "mode"
   GENERATOR_POINTS = "generatorPoints"
   OPTIMIZER = "optimizer"
   NOISE = "noise"

   def __str__(self):
      return str(self.value)