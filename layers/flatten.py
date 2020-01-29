
# import standard modules

# import third party modules
import numpy as np

# import project related modules


class Flatten:

    def __init__(self):
        self.cache = None

    def forward(self, x):

        # save input for backward reshaping
        self.cache = x
        return x.flatten(order='C')

    def backward(self, da):
        return da.reshape(self.cache.shape)