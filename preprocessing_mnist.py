import pickle
import os
import numpy as np
# https://www.python-course.eu/neural_network_mnist.php

# load mnist data set for configuration
path = "/Users/maximperl/Downloads"
train_data = np.loadtxt(os.path.join(path, "mnist_train.csv"), delimiter=",")

# Map pixel values into an interval from [0.01, 1] by multiplying each pixel by 0.99 / 255 and adding 0.01 to
# the result. This way, we avoid 0 values as inputs, which are capable of preventing weight updates, as we we seen
# in the introductory chapter.
fac = 0.99 / 255
train_imgs = np.asfarray(train_data[:, 1:]) * fac + 0.01
train_labels = np.asfarray(train_data[:, :1])
classes = 10

# one hot encode labels
lr = np.arange(10)
lr = np.arange(classes)

for label in range(10):
    one_hot = (lr == label).astype(np.int)
r = np.arange(classes)

# transform labels into one hot representation
train_labels_one_hot = (lr == train_labels).astype(np.float)

# we don't want zeroes and ones in the labels neither:
train_labels_one_hot[train_labels_one_hot == 0] = 0.01
train_labels_one_hot[train_labels_one_hot == 1] = 0.99

# transform training data into a image shape (28, 28, 1)
X_train = np.zeros((train_imgs.shape[0], 28, 28))
for x in range(train_imgs.shape[0]):
    X_train[x] = train_imgs[x].reshape(28, 28)

# dump data for faster reload

with open("mnist.pk1", "bw") as fh:
    data = (
        X_train,
        train_labels,
        train_labels_one_hot
    )
    pickle.dump(data, fh)