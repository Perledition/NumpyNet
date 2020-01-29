
# import standard modules

# import third party modules
import numpy as np

# import project related modules
from _meta.layer import Layer


# TODO: insert border modes
# TODO: add kernel dimensions to the layer
class ConvolutionalLayer(Layer):

    def __init__(self, filter_size=3, filter_count=3, learning_rate=0.01, layer_id=None, border_mode='valid'):

        super().__init__(layer_id)

        assert border_mode in ['half', 'same', 'valid', 'zero'],\
            "border_mode is not supported please choose half, same, valid or zero"  # quality check of border mode

        # learning rate to update the filters in backward process
        self.learning_rate = learning_rate
        self.border_mode = border_mode

        # general information about the filter size and the amount of filters used
        self.filter_size = filter_size
        self.filter_count = filter_count

        if self.initialize_filters:
            self.filters = np.random.randn(self.filter_count, self.filter_size, self.filter_size) \
                           / (self.filter_size * self.filter_size)  # divide by power filter size to reduce the variance
        else:
            self.filters = self.load_weights()

    def window_operation(self, image):
        """
        calculate all positions given for a window hover operation and create a generator out of it.
        uses valid padding by default
        :param image: (numpy.array): input image to process during a hover operation
        :return: generator: all positions for a filter window to pass
        """
        height, width = image.shape
        for row in range(height - 2):
            rs = row
            re = rs + self.filter_size
            for column in range(width - 2):
                cs = column
                ce = cs + self.filter_size
                window = image[rs:re, cs:ce]
                yield window, row, column

    def forward(self, x):

        # store the input for later backward processing
        self.cache = x.copy()

        # get amount of rows and columns from input data to calculate output dimensions
        print(x.shape)
        height, width = x.shape

        # initialize an array for data output
        result = np.zeros((height - 2, width - 2, self.filter_count))

        for window, row, column in self.window_operation(x):
            result[row, column] = np.sum(window * self.filters, axis=(1, 2))

        return result

    def backward(self, dl):

        d_filters = np.zeros(self.filters.shape)

        for window, row, column in self.window_operation(self.cache):
            for kernel in range(self.filter_count):
                d_filters[kernel] += dl[row, column, kernel] * window

        # update filters with the given loss w.r.t to filters
        self.filters -= self.learning_rate * d_filters

        return d_filters
