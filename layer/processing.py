# import standard modules
import numpy as np


class Sequential:

    def __init__(self, layers):
        self.layers = layers

    def train(self, x):
        for layer in self.layers:
            x = layer.assign(x)

