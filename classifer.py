# import standard modules

# import third party modules
import numpy as np
import sklearn.datasets as skd
from matplotlib import pyplot as plt

# project related imports
from layers.connected import Dense
from layers.activations import Sigmoid, RelU
from model.model import Sequence


# network assumes quadratic format of input
# x, y = PreProcessing(r'C:\Users\Maxim\Documents\Studium\Semester\Semester I\DataScience\Studienleistung\cifar-10-batches-py').load_all()

# create test data

def load_extra_datasets():
    N = 1000
    gq = skd.make_gaussian_quantiles(mean=None, cov=0.7, n_samples=N, n_features=2, n_classes=2,  shuffle=True, random_state=None)
    return gq


gaussian_quantiles= load_extra_datasets()
X, Y = gaussian_quantiles
X, Y = X, Y.reshape(-1, 1)
Y = np.append(Y, np.where(Y==1, 0, 1), axis=1)
print(X.shape, Y.shape)

# initialize the model
model = Sequence([
    Dense(8, 2, input_layer=True),
    RelU(),
    Dense(16, 8),
    Sigmoid(),
    Dense(2, 16, activation="softmax"),
],
    loss="cross_entropy",
    epochs=250,
    batch_size=1

)

# train the model on the given data
model.fit(X, Y)


# plot loss
plt.plot([x for x in range(0, len(model.loss_history))], model.loss_history)
plt.show()