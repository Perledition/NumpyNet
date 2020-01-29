# import standard modules
import uuid

# import third party modules
import numpy as np


# TODO: write function to save/load filters automatically
class Layer:

    def __init__(self, layer_id):

        # cache will be assign with the input during the forward process to save it for the backward operation
        self.cache = None
        self.output = None

        if layer_id is None:
            self.layer_id = uuid.uuid1()
            self.initialize_filters = True
        else:
            self.layer_id = layer_id
            self.initialize_filters = False

    def load_weights(self):
        x = self.layer_id
        return 1, 1

    def write_weights(self):
        x = self.layer_id
        return 1, 1
