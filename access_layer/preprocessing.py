# import standard modules
import pickle
import os.path
import logging

# import third party modules
import numpy as np


class PreProcessing:
    """
    PreProcessing is oriented on Dataset Layout of:
    http://www.cs.toronto.edu/~kriz/cifar.html

    Because this data was used for the classification
    Loaded in this way, each of the batch files contains a dictionary with the following elements:

    data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image.
    The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue.
    The image is stored in row-major order, so that the first 32 entries of the array are the red channel values
    of the first row of the image.

    labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the
    ith image in the array data.


    The dataset contains another file, called batches.meta.
    It too contains a Python dictionary object. It has the following entries:

    label_names -- a 10-element list which gives meaningful names to the numeric labels in the labels array
    described above. For example, label_names[0] == "airplane", label_names[1] == "automobile", etc.

    """

    def __init__(self, location):
        assert os.path.exists(location), "given path folder path is not valid for image data, try to use r'string'"
        self.folder_location = location
        self.x = list()
        self.y = list()

    def get_files(self):

        # create file paths
        files = [f for f in os.listdir(self.folder_location) if f.startswith('data_batch')]
        files = [os.path.join(self.folder_location, f) for f in files]

        return files

    @staticmethod
    def unpickle(path):
        with open(path, 'rb') as fo:
            data_dict = pickle.load(fo, encoding='bytes')
        return data_dict

    def load_single(self, path):
        data = self.unpickle(path)

        array_data = data[b'data']
        labels = data[b'labels']

        # split the data into three channels and transform its shape to 32x32 image - cut is at 1024 columns
        # each row contains one training sample
        for row in range(0, 10000):
            sample = array_data[row]

            # create empty image array
            rgbArray = np.zeros((32, 32, 3), 'uint8')

            # add red channel from sample data
            rgbArray[:, :, 0] = sample[:1024].reshape((32, 32))

            # add green channel from sample data
            rgbArray[:, :, 1] = sample[1024:2048].reshape((32, 32))

            # add blue channel from sample data
            rgbArray[:, :, 2] = sample[2048:3072].reshape((32, 32))

            # add image and label to data storage
            self.x.append(rgbArray)
            self.y.append(labels[0])

        return True

    def load_all(self):

        for file in self.get_files():
            self.load_single(file)

        return self.x, self.y
