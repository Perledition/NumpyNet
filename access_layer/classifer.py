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

# train test split for data
X_train = x[:400]
y_train = y[:, :400]

X_test = x[-100:]
y_test = y[:, -100:]


model = Sequential(
    [
        ConvolutionalLayer(),
        PoolingLayer(),
        Flatten(),
        DenseLayer(nodes=10, output_layer=True, activation_type='softmax')
    ],
    epochs=100
)

model.train(X_train, y_train)

