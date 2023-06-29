from numpy import array, concatenate
from PIL import Image
import json
from os import listdir
from os.path import isfile, join
from constants import FILES, LAYERS, ConfigOptions
from config import Config

def parseInput(filepath):
    with open(filepath) as f:
        lines = f.readlines()
        data = []
        for line in lines:
            data.append(array([1.0] + [float(elem) for elem in line.strip().split()]))
    return array(data)

def parseImages(directory):
    data = []
    size = None
    images = [join(directory, f) for f in listdir(directory) if isfile(join(directory, f))]
    for image in images:
        img = Image.open(image)
        size = img.size
        greyscaleImg = img.convert(mode="1", dither=Image.NONE)
        grayscalePixels = array(greyscaleImg.getdata())/255
        data.append(concatenate((array([1]), grayscalePixels)))
    return array(data), size

def parseConfiguration(configPath):
    with open(configPath) as json_file:
        data = json.load(json_file)
        
        files = data[FILES]
        layers = data[LAYERS]
        inputData = files[ConfigOptions.INPUT_DATA.value]
        iterations = data[ConfigOptions.ITERATIONS.value]
        learningRate = data[ConfigOptions.LEARNING_RATE.value]
        momentum = data[ConfigOptions.MOMENTUM.value]        
        error = data[ConfigOptions.ERROR_LIMIT.value]
        beta = data[ConfigOptions.BETA.value]
        alpha = data[ConfigOptions.ALPHA.value]
        mode = data[ConfigOptions.MODE.value]
        generatorPoints = data[ConfigOptions.GENERATOR_POINTS.value]
        optimizer = data[ConfigOptions.OPTIMIZER.value]     
        noise = data[ConfigOptions.NOISE.value]  
        
        config = Config(
            inputs=inputData,
            iterations=iterations,
            learningRate=learningRate,
            momentum=momentum,
            error=error,
            layers=layers,
            beta=beta,
            alpha=alpha,
            mode=mode,
            generatorPoints=generatorPoints,
            optimizer=optimizer,
            noise=noise
        )
    return config