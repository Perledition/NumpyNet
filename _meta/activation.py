# import standard modules
import math

# import third party modules
import numpy as np


class Softmax:

    @staticmethod
    def forward(x):
        # this function will calculate the probabilities of each
        # target class over all possible target classes

        # get max values and shape reshape them for substraction
        maxes = np.amax(x, axis=1)
        maxes = maxes.reshape(maxes.shape[0], 1)

        # calculate exp value for input minues max values
        e = np.exp(x - maxes)

        # create probability distribution for all output classes
        dist = e / np.sum(e, axis=1, keepdims=True)
        return dist

    @staticmethod
    def backward(da, cache):
        return da * cache * (1 - cache)


class Sigmoid:

    @staticmethod
    def forward(x):
        # apply features to sigmoid function
        return 1.0 / (1.0 + math.exp(-x))

    def backward(self, da, cache):
        z = self.forward(cache)
        return z * (1 - z) * da


class ReLU:

    @staticmethod
    def forward(x):
        # initialize an array in shape of input with zeros
        z = np.zeros_like(x)
        # ReLU activation -> return every pixel/ feature above zero
        return np.where(x > z, x, z)

    @staticmethod
    def backward(da, cache):
        # multiply all positive values with gradient greater than zero or assign zero to the position
        return da * np.where(cache >= 0, 1, 0)