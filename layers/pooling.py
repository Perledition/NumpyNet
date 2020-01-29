# import standard modules

# import third party modules
import numpy as np

# import project related modules
from _meta.layer import Layer


class PoolingLayer(Layer):

    def __init__(self, pool_size=2, pooling_type='max', layer_id=None):
        super().__init__(layer_id)
        self.pool_size = pool_size
        self.pooling_type = pooling_type
        self.pooling = {'max': np.max, 'min': np.min, 'mean': np.mean}[pooling_type]

    def window_operation(self, image):
        """
        calculate all positions given for a window hover operation and create a generator out of it.
        uses valid padding by default
        :param image: (numpy.array): input image to process during a hover operation
        :return: generator: all positions for a filter window to pass
        """
        height, width = image.shape
        for row in range(height//self.pool_size):
            rs = row * self.pool_size
            re = rs + self.pool_size
            for column in range(width//self.pool_size):
                cs = column * self.pool_size
                ce = cs + self.pool_size
                window = image[rs:re, cs:ce]
                yield window, row, column

    def forward(self, x):
        self.cache = x.copy()  # write input data into cache to store it for back prop operation

        # get dimensions of input data
        height, width, kernel = x.shape
        result = np.zeros((int(height/self.pool_size), int(width/self.pool_size), kernel))

        # run window hover operation to extract most relevant features based on pooling type
        for k in range(kernel):
            for window, row, column in self.window_operation(x[:, :, k]):
                result[row, column, k] = self.pooling(window)

        return result

    def backward(self, da):

        # initialize an empty array as output matrix which will be filled during the back propagation operation
        dl_cache = np.zeros(self.cache.shape)

        # perform window operation and pull max features
        for window, row, column in self.window_operation(self.cache):
            height, width, kernels = window.shape
            pixel_value = self.pooling(window)

            for row_back in range(height):
                for column_back in range(width):
                    for kernel_back in range(kernels):

                        # if current pixel value matches the max / min value of previous forward operation assign
                        # gradient to it
                        if window[row_back, column_back, kernel_back] == pixel_value:

                            # define index values to assign value
                            r = row * self.pool_size + row_back
                            c = column * self.pool_size + column_back

                            #  copy gradient data to position where it matches
                            dl_cache[r, c, kernel_back] = da[row, column, kernel_back]

        return dl_cache
