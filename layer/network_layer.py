# import standard modules
import os
import math
import uuid

# import third party modules
import numpy as np


class PoolingLayer:
    """
    Assumes square format
    """

    def __init__(self, pool_size=2, pooling_type='max'):
        self.pool_size = pool_size
        self.pooling = {'max': np.max, 'min': np.min, 'mean': np.mean}[pooling_type]

    def pool_2d(self, feature):

        # create a temp array to save the resulting calculations
        channels, width, height = feature.shape
        result = np.zeros((channels, int(width/self.pool_size), int(height/self.pool_size)))

        # apply window slicing process over all three channels with a kernel for each channel
        for row in range(0, int(height/self.pool_size)):
            rs = 0 + row  # starting point in row for this window
            re = self.pool_size + row  # endpoint in row for this window

            for column in range(0, int(width/self.pool_size)):
                cs = 0 + column  # starting point in column for this window
                ce = self.pool_size + column  # endpoint in column for this window

                for channel in range(0, channels):
                    # assign the max, min, avg value of the current window to the resulting array space
                    result[channel, row, column] = self.pooling(feature[channel, rs:re, cs:ce])

        return np.array(result)

    def assign(self, x):

        # get current dimensions of the image features and divide it's width and height be pool size to get output dim's
        resulting_dim = list(x.shape)
        resulting_dim[-2:] = [int(resulting_dim[-2]/self.pool_size), int(resulting_dim[-2]/self.pool_size)]

        # initialize our more dense feature list as empty array
        pool_features = np.zeros(tuple(resulting_dim))

        # iterate over each sample given in x (sample, tensor_dimension, channels, height, width)
        for idx, sample in enumerate(x):
            # iterate over each tensor dimension (tensor_dimension, channels, height, width)
            for tdx, tensor_dim in enumerate(sample):
                pool_features[idx, tdx] = self.pool_2d(tensor_dim)

        # run process for each image in feature and return the filled feature array
        return pool_features


class ActivationLayer:

    def __init__(self, activation_type='relu'):
        self.activation_type = activation_type

        self.activation = {'relu': self.relu_layer,
                           'sigmoid': self.sigmoid_layer,
                           'softmax': self.softmax_layer}

    @staticmethod
    def relu_layer(x):
        # turn all negative values in a matrix into zeros
        z = np.zeros_like(x)
        return np.where(x > z, x, z)

    @staticmethod
    def sigmoid_layer(x):
        return 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def softmax_layer(w):
        # this function will calculate the probabilities of each
        # target class over all possible target classes.
        maxes = np.amax(w, axis=1)
        maxes = maxes.reshape(maxes.shape[0], 1)
        e = np.exp(w - maxes)
        dist = e / np.sum(e, axis=1, keepdims=True)
        return dist

    def assign(self, x):
        return self.activation[self.activation_type](x)


class DenseLayer:

    def __init__(self, layers, index):
        self.layers = layers
        self.index = index

    def assign(self, x):
        w = self.layers[self.index]['parameter_0']
        b = self.layers[self.index]['parameter_1']
        return np.dot(x, w) + b


class DropoutLayer:
    # affect the probability a node will be turned off by multiplying it
    # by a p values (.25 we define)

    def __init__(self, prob=.25):
        self.prob = prob

    def assign(self, x):
        retain_prob = 1. - self.prob
        x *= retain_prob
        return x


class ConvolutionalLayer:

    def __init__(self, border_mode='valid', filter_size=3, filter_count=32, layer_id=None):

        # make border mode a global variable of the class and do quality check beforehand
        # With border mode "valid"/"zero" you get an output that is smaller than the input because the convolution is
        # only computed where the input and the filter fully overlap.

        assert border_mode in ['half', 'same', 'valid', 'zero'],\
            "border_mode is not supported please choose half, same, valid or zero"  # quality check of border mode

        self.border_mode = border_mode  # defines padding for input data
        self.filter_count = filter_count  # defines how many filters will be initialized for this Layer
        self.filter_size = filter_size  # defines the shape of the one filter e.g. size 3 means a 3x3 kernel/filter

        # assign random layer id code to the layer to make temp files and results locatable
        if layer_id is None:
            self.layer_id = uuid.uuid1()
            self.kernels = self.initialize_filters()

        # assign known layer id in case structure is reused for prediction or to assign pre trained filters
        else:
            self.layer_id = layer_id
            self.kernels = self.load_filters()

    def conv_2d(self, x):
        """
        Takes a data batch and a kernel an creates for each sample a channel x width x height array
        therefore a output of batch_size x channel x width - 2 x height - 2 array as convolved feature
        :param data: batch
        :param kernel: filter to be applied to the data - has to be one filter at a time
        :return:
        """

        # create 0 padding around the given data
        if self.border_mode in ['half', 'same']:

            # get shape of current input
            c, h, w = x.shape

            # create a temp zero filled array with an padding of one on each side
            padded_array = np.zero((c, h + 2, w + 2))

            # assign the real data of each channel into the padded_array
            for i in range(3):
                padded_array[i, 1:h+1, 1:w+1] = x

            data = padded_array

        # if no padding assign variable data with x
        else:
            data = x

        conv_features = [0 for _ in range(0, len(self.kernels))]

        # iterate through all 32 filters and apply it on the given sample
        for idx, kernel in enumerate(self.kernels):

            # create a temp array to save the resulting calculations
            channels, width, height = data.shape
            result = np.zeros((channels, width - 2, height - 2))

            # apply window slicing process over all three channels with a kernel for each channel
            for row in range(0, height-2):
                rs = 0 + row  # starting point in row for this window
                re = self.filter_size + row  # endpoint in row for this window

                for column in range(0, width-2):
                    cs = 0 + column  # starting point in column for this window
                    ce = self.filter_size + column  # endpoint in column for this window

                    for channel in range(0, channels):
                        # assign sum of multiplication process to the resulting array
                        # Mathematically, it’s (2 * 1) + (0 * 0) + (1 * 1) = 3 TODO: implement real formula
                        result[channel, row, column] = np.sum(data[channel, rs:re, cs:ce] * kernel[channel])

            conv_features[int(idx)] = result

        return np.array(conv_features)

    # kernel and filters are the same
    def initialize_filters(self):
        # create random filter shapes by default 32 filters with a size of 3 channels x 3 width x 3 height
        # the cnn assumes always 3 channels
        kernels = [np.random.randint(-1, 2, size=(3, self.filter_size, self.filter_size))
                   for _ in range(32)]

        # save a temp file in which the filters will be saved for later usage
        self.write_filters(kernels)
        return kernels

    def load_filters(self):

        # create the path to load from and load a npy file with the kernels
        path = '/'.join(folder for folder in os.getcwd().split('\\')[:-1])
        path = os.path.join(path, f'temps/temp_{self.layer_id}.npy')
        return np.load(path)

    def write_filters(self, filters):

        # create the right path to save the data to and save kernels as npy file
        path = '/'.join(folder for folder in os.getcwd().split('\\')[:-1])
        path = os.path.join(path, f'temps/temp_{self.layer_id}.npy')
        np.save(path, filters)

    def assign(self, x):
        """
        assign function is the main loop for the ConvolutionalLayer class and will be the only public used function
        from other classes like Sequential. The function is designed to process a batch of 3d arrays with shape like:
        (samples, channels, width, height). However, to use assign accordingly x must be at least a single 3d array.
        The function will iterate through each sample and apply the conv_2d method for each sample with each filter.

        :param x: (np.array): at least 3d numpy array with training data, 4d recommended: collection of 3d arrays
        :return: (np.array): resulting convolved features for each sample provided to assign
        """

        # initialize placeholder variable for returning values
        conv_features = None

        # iterate over each sample and apply convolution
        for nr, sample in enumerate(x):

            # initialize placeholder variable for an array which will hold all applied filter operations for one sample
            kernel_run_features = None

            # add a kernel_dimension of one in case it's the first convolution iteration
            if len(sample.shape) < 4:
                sample = np.array([sample])

            # loop over each dimension of the given sample
            # in case of single image this is one (1, channels, width, height) but with 32 filters applied in a previous
            # step it would be a tensor of (32, channels, width, height)
            for tensor_dimension in sample:

                if kernel_run_features is None:
                    # gives (32, channels, width, height)
                    kernel_run_features = self.conv_2d(tensor_dimension)

                else:
                    # return (32, channels, width, height)
                    features = self.conv_2d(tensor_dimension)

                    # concat features so the output for the example above would be (64, channels, width, height)
                    kernel_run_features = np.append(kernel_run_features, features, axis=0)

            if conv_features is None:
                # return kernel_count x kernel_count, 3, 30, 30
                conv_features = np.array([kernel_run_features])

            else:
                # return kernel_count x kernel_count, 3, 30, 30
                conv_features = np.append(conv_features, [kernel_run_features], axis=0)

            # status update
            if nr == len(x) - 1:
                print(f'Conv: {"="*100 + ">"}|{nr}/{len(x)}')
            else:
                print(f'Conv: {"="*(100 - int((1 - (nr + 1/len(x)))*100))+ ">" +  " "*int((1 - (nr + 1/len(x)))*100)}|{nr+1}/{len(x)}')

        return conv_features


class Flatten:
    def assign(self, x):
        flat_object = np.zeros((x.shape[0], np.prod(x.shape[1:])))
        for i in range(x.shape[0]):
            flat_object[i, :] = x[i].flatten(order='C')
        return flat_object
