# import standard modules
import math

# import third party modules
import numpy as np


class PoolingLayer:
    """
    Assumes square format
    """

    def __init__(self, features, pool_size=2, pooling_type='max'):
        self.pool_size = pool_size
        self.features = features
        self.nearby_features, self.nearby_images, self.feature_dimension = features.shape
        self.resulting_dimension = int(self.feature_dimension / pool_size)
        self.pooling = {'max': np.max, 'min': np.min}[pooling_type]

    def feature_loop(self, pooling_results):

        # execute the max pooling process for each image in handed over features
        for image in range(self.nearby_images):

            # for each channel in an image array
            for feature in range(self.nearby_features):

                # hovering or window sliding process is defined by rows and columns to assign
                for row in range(self.resulting_dimension):

                    # define positions on row level
                    row_starting_point = row * self.pool_size
                    row_end_point = row + self.pool_size

                    for col in range(self.resulting_dimension):

                        # define positions on column level
                        col_starting_point = col * self.pool_size
                        col_end_point = col + self.pool_size

                        # based on rows, columns, image and channel extract a patch of values from the image
                        patch = self.features[feature, image,
                                              row_starting_point:row_end_point,
                                              col_starting_point:col_end_point]

                        # reduce patch by type and assign result to output matrix
                        pooling_results[feature, image, row, col] = self.pooling(patch)

        return pooling_results

    def assign(self):
        # initialize our more dense feature list as empty array
        pool_features = np.zeros((self.nearby_features, self.nearby_images,
                                  self.feature_dimension, self.feature_dimension))

        # run process for each image in feature and return the filled feature array
        return self.feature_loop(pool_features)


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

    def __init__(self, layers, index=0, border_mode='full'):
        self.border_mode = border_mode
        self.layers = layers
        self.index = index

    def conv_2d(self, image, feature):
        image_dim = np.array(image.shape)
        feature_dim = np.array(feature.shape)
        target_dim = image_dim + feature_dim - 1
        operation_result = np.fft.fft2(image, target_dim) * np.fft.fft2(feature, target_dim)
        target = np.fft.ifft2(operation_result).real

        if self.border_mode is 'valid':
            # To compute a valid shape, either np.all(x_shape >= y_shape) or
            # np.all(y_shape >= x_shape).
            valid_dim = image_dim - feature_dim + 1
            if np.any(valid_dim < 1):
                valid_dim = feature_dim - image_dim + 1
            start_i = (target_dim - valid_dim) // 2
            end_i = start_i + valid_dim
            target = target[start_i[0]:end_i[0], start_i[1]:end_i[1]]
        return target

    def assign(self, x):
        features = self.layers[self.index]['parameter_0']
        bias = self.layers[self.index]['parameter_1']

        patch_dim = features[0].shape[-1]
        nearby_features = features.shape[0]
        image_dim = x.shape[2]  # assume image square
        channels = x.shape[1]
        nearby_images = x.shape[0]

        assert self.border_mode in ['valid', 'full'], f'border mode {self.border_mode} is not supported. valid or full'

        if self.border_mode is 'full':
            conv_dim = image_dim + patch_dim - 1

        elif self.border_mode is 'valid':
            conv_dim = image_dim - patch_dim + 1

        else:
            conv_dim = image_dim

        results = np.zeros((nearby_images, nearby_features, conv_dim, conv_dim))

        for image in range(nearby_images):
            for feature in range(nearby_features):

                conv_image = np.zeros((conv_dim, conv_dim))

                for channel in range(channels):

                    feature = features[feature, channel, :, :]
                    image = x[image, channel, :, :]
                    conv_image += self.conv_2d(image, feature)

                conv_image = conv_image + bias[feature]
                results[image, feature, :, :] = conv_image
        return results


class Flatten:
    def assign(self, x):
        flat_object = np.zeros((x.shape[0], np.prod(x.shape[1:])))
        for i in range(x.shape[0]):
            flat_object[i, :] = x[i].flatten(order='C')
        return flat_object
