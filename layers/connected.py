# import standard modules

# import third party modules
import numpy as np

# import project related modules
from _meta.layer import Layer
from .activation import ActivationLayer


class DenseLayer(Layer):

    def __init__(self, nodes=10, output_layer=False, activation_type='relu', layer_id=None, learning_rate=0.01):
        super().__init__(layer_id)
        self.learning_rate = learning_rate
        self.nodes = nodes
        self.output_layer = output_layer
        self.activation_type = activation_type
        self.weights, self.bias = None, None
        self.activation = ActivationLayer(self.activation_type)
        self.z = None

    def initialize_weights(self, input_dim):
        weights = np.random.rand(self.nodes, input_dim) * np.sqrt((2/input_dim))
        bias = np.zeros((self.nodes, 1))
        return weights, bias

    def forward(self, x):

        # copy current input as cache for back propagation
        self.cache = x.copy()

        # check whether new weights must be initialized or if one could pre trained filters
        if self.initialize_filters:
            self.weights, self.bias = self.initialize_weights(x.shape[0])
        else:
            self.weights, self.bias = self.load_weights()

        # calculate the output matrix with given weights and bias
        self.z = np.dot(self.weights, x.transpose()) + self.bias

        # apply choosen activation function on output data and return it
        # save output in variable to be able to use it for back propagation operation
        self.output = self.activation.forward(self.z)
        return self.output

    def backward(self, da, y=None):

        # output layer error
        if self.output_layer:

            # calculate layer error
            error = (self.output - y) * self.activation.backward(da)

        # hidden layer error
        else:
            error = da * self.weights * self.activation.backward(da)

        # cost derivative for weights and bias
        dw = 1/self.cache.shape[0] * np.dot(error * self.cache)
        db = 1/self.cache.shape[0] * np.sum(error, axis=1, keepdims=True)

        # update weights
        self.weights = self.weights - self.learning_rate * dw
        self.bias = self.bias - self.learning_rate * db

        return error



