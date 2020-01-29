# import standard modules

# import third party modules+

# project related imports
from preprocessing import PreProcessing
from layers.pooling import PoolingLayer
from layers.flatten import Flatten
from layers.connected import DenseLayer
from layers.convolution import ConvolutionalLayer
from model.model import Sequential


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
    epochs=1000
)

model.train(X_train, y_train)