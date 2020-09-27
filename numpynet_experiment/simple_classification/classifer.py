# import standard modules

# import third party modules
import numpy as np
import sklearn.datasets as skd
from matplotlib import pyplot as plt

# project related imports
from model.model import Sequence
from layers.connected import Dense
from layers.activations import Sigmoid, RelU
from metric.measure import ClassificationMatrix

################################
#      setup create data       #
################################


# function creates a data set with sci-kit-learn this is the only part where another module beside numpy and pythons
# built in packages are used in order to make the data creation easy since this is not was the repository is about
def generate_data():
    """
    creates a simple 2d data set with two classes. The data set has a circle of class one and within the circle the
    points belong to class 2. This is a pretty well known data set configuration and ideal for a short tryout.

    :return: tuple: x data consisting of x1 and x2 variable and y data as class / label / target for the data set
    """
    N = 1000
    gq = skd.make_gaussian_quantiles(mean=None, cov=0.7, n_samples=N, n_features=2, n_classes=2,
                                     shuffle=True, random_state=None)
    return gq


# create the data set
X, Y = generate_data()

# reshape the data and make Y a one hot encoded vector in order to test multi-output with Numpy Net
X, Y = X, Y.reshape(-1, 1)
Y = np.append(Y, np.where(Y == 1, 0, 1), axis=1)

# print dimensions of x and y values
print(X.shape, Y.shape)

# split in train and test
X_train = X[:900, :]
X_test = X[900:, :]

Y_train = Y[:900, :]
Y_test = Y[900:, :]


################################
#         setup model          #
################################

# create the list directly with in the Sequence structure to see the layers created and their order more easily
# The layers are exchangeable and the parameters can be modified as well, so try to play around with it :)
model = Sequence([
    Dense(8, input_layer=True),
    RelU(),
    Dense(16),
    Sigmoid(),
    Dense(2, activation="softmax"),
],
    loss="cross_entropy",
    feedback_steps=10,
    epochs=70,

)

# comment in in order to see print the model architecture to the console
model.show()

# plot loss curve if of interest
# plt.plot([x for x in range(0, len(model.loss_history))], model.loss_history)
# plt.title("loss curve of simple classification problem")
# plt.ylabel("loss")
# plt.xlabel("epochs")
# plt.show()

# train the model on the given data
model.train(X_train, Y_train)


################################
#          test model          #
################################

# set up a classification matrix and test whether the model is over fits.
print("\nClassification Matrix with Y as Rows and Predicted as Columns")
predictions = model.predict(X_test)
matrix = ClassificationMatrix().fit(predictions, Y_test)
print(matrix)
