# import standard modules

# import third party modules
import numpy as np

# import project related modules


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
