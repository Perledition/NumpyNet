# import standard modules

# import third party modules
import numpy as np
from matplotlib import pyplot as plt

# project related imports
from model.model import Sequence
from layers.connected import Dense
from layers.activations import Sigmoid, RelU
from metric.measure import ClassificationMatrix

from numpynet_experiment.mnist_digit_classification.decoding import decode_mnist

################################
#      setup create data       #
################################

# get the data set from: https://www.python-course.eu/neural_network_mnist.php since it does not make sense to set the
# data with the code. How ever, once you load the data replace the current path for np.loadtxt functions with out paths
train_data = np.loadtxt("C:\\Users\Maxim\Downloads\mnist_train(1).csv", delimiter=",")

# create x and y data from the training set the function comes from decoding.py
X_train, y_train = decode_mnist(train_data)

################################
#         setup model          #
################################

model = Sequence([
    Dense(785, input_layer=True),
    RelU(),
    Dense(160),
    Sigmoid(),
    Dense(10, activation="softmax"),
],
    loss="cross_entropy",
    feedback_steps=10,
    epochs=200,
)

print("start training")

# use only 2000 examples in order to save time
model.train(X_train[:2000], y_train[:2000])


# plot loss curve if of interest
# plt.plot([x for x in range(0, len(model.loss_history))], model.loss_history)
# plt.title("loss curve of simple classification problem")
# plt.ylabel("loss")
# plt.xlabel("epochs")
# plt.show()

################################
#          test model          #
################################

# set up a classification matrix and test whether the model is over fits.
print("\nClassification Matrix with Y as Rows and Predicted as Columns")

# take 500 unused samples from the data set in order to test the performance
predictions = model.predict(X_train[2000:2500])
matrix = ClassificationMatrix().fit(predictions, y_train[2000:2500])
print(matrix)