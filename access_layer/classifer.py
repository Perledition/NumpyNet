# import standard modules
import pickle

# import third party modules+
import numpy as np

# project related imports
from access_layer.preprocessing import PreProcessing
from layer.processing import Sequential
from layer.network_layer import *


class NumpyObjectClassifier:

    def __init__(self, version='latest', pool_size=2):
        super().__init__(version)
        self.weights = pickle.load(open(self.stack_path('model', f'weights_{version}.pk1')))
        self.pool_size = pool_size

    def transform_predict(self):
        pass

    def transform_retrain(self):
        pass


x, y = PreProcessing(r'C:\Users\Maxim\Downloads\cifar-10-batches-py').load_all()

model = Sequential([
    ConvolutionalLayer(),
    ActivationLayer(),
    ConvolutionalLayer(),
    ActivationLayer(),
    PoolingLayer(),
    DropoutLayer(),
    Flatten(),
    DenseLayer(),
    ActivationLayer(),
    DropoutLayer(),
    DenseLayer(),
    ActivationLayer(activation_type='softmax')
]).train(x[0])

print(model)