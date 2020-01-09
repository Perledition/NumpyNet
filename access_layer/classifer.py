# import standard modules
import pickle

# import third party modules+
import numpy as np

# project related imports
from access_layer.preprocessing import PreProcessing
from layer.processing import Sequential
from layer.network_layer import *


# network assumes quadratic format of input
x, y = PreProcessing(r'C:\Users\Maxim\Documents\Studium\Semester\Semester I\DataScience\Studienleistung\cifar-10-batches-py').load_all()

"""
model = Sequential(
    [
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
    ],
    epochs=100
)

model.train(x[0])
"""

print(x[0].shape)
cl = ConvolutionalLayer().assign(x[0:2])
cl = ActivationLayer().assign(cl)
cl = ConvolutionalLayer().assign(cl)
cl = ActivationLayer().assign(cl)
print(cl.shape)
cl = PoolingLayer().assign(cl)
print(cl.shape)
cl = Flatten().assign(cl)
cl = DenseLayer().assign(cl)
cl = DenseLayer(activation_type='softmax').assign(cl)
print(cl)