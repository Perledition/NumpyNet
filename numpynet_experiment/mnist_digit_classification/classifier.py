# import standard modules

# import third party modules
import numpy as np
from matplotlib import pyplot as plt

# project related imports
from model.model import Sequence
from layers.connected import Dense
from layers.activations import Sigmoid, RelU
from metric.measure import ClassificationMatrix

################################
#      setup create data       #
################################

# get the data set from: https://www.python-course.eu/neural_network_mnist.php since it does not make sense to set the
# data with the code. How ever, once you load the data replace the current path for np.loadtxt functions with out paths
train_data = np.loadtxt("C:\\Users\Maxim\Downloads\mnist_train(1).csv", delimiter=",")


def decode_mnist(data):
    """
    the logic of how to encode the data set properly comes from the source and is mainly used in as it is described in
    order to save time and make the data set quickly available for a test run.

    :param data: mnist dataset to be modified into usable classification data
    :return: tuple: array with feature data of x and one hot encoded y as labels
    """

    # factor for data to normalize the data and make computation more efficient
    fac = 0.99 / 255

    # convert data set into array and normalize it's pixel values
    images = np.asfarray(data[:, 1:]) * fac + 0.01

    # convert labels into array and one hot encode the labels accordingly - 10 digits classes to train on
    labels = np.asfarray(data[:, :1])
    unique_classes = np.unique(labels).shape[0]
    label_one_hot = np.zeros((labels.shape[0], unique_classes))

    for r in range(labels.shape[0]):
        index = int(labels[r][0])
        label_one_hot[r, index] = 1

    return images, label_one_hot


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
predictions = model.predict(X_train[2000:2500])
matrix = ClassificationMatrix().fit(predictions, y_train[2000:2500])
print(matrix)