# import standard modules
import numpy as np


class Sequential:

    def __init__(self, layers, epochs=100):
        self.layers = layers
        self.epochs = epochs
        self.crossentropy = None

    def print_process(self):
        pass

    def batch_generator(self):
        pass

    def train(self, x):
        for layer in self.layers:
            x = layer.assign(x)

