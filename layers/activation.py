# import standard modules

# import third party modules

# import project related modules
from _meta.activation import *


class ActivationLayer:

    def __init__(self, activation_type='relu'):
        self.activation_type = activation_type
        self.cache = None

        # create dict with activation functions classes to choose from
        self.activation = {
            'relu': ReLU(),
            'sigmoid': Sigmoid(),
            'softmax': Softmax(),
        }

    def forward(self, x):

        # save input data for back propagation
        self.cache = x.copy()

        # run forward propagation method with input for given activation function class
        return self.activation[self.activation_type].forward(x)

    def backward(self, da):

        # run backward operation for activation layer and handover cache from forward operation next to gradient
        return self.activation[self.activation_type].backward(da, self.cache)
