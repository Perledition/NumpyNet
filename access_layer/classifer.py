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


model = Sequential(
    [
        ConvolutionalLayer(),
        PoolingLayer(),
        ActivationLayer(),
        ConvolutionalLayer(),
        PoolingLayer(),
        ActivationLayer(),
        # DropoutLayer(),
        Flatten(),
        DenseLayer(nodes=256),
        # DropoutLayer(),
        DenseLayer(nodes=100),
        DenseLayer(nodes=10, output_layer=True, activation_type='softmax')
    ],
    epochs=100
)

model.train(x[:100], y[:, :100])

