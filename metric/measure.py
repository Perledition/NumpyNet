# import standard modules

# import third party modules
import numpy as np

# import project related modules


class Metrics:

    def __init__(self):
        pass


class Recall(Metrics):

    def __init__(self):
        super().__init__()


class Accuracy(Metrics):

    def __init__(self):
        super().__init__()
        self.values = list()

    def add(self, x, y):
        pred = np.argmax(x)
        target = np.argmax(y)

        if int(pred)-int(target) == 0:
            self.values.append(1)
        else:
            self.values.append(0)

    def clean(self):
        self.values = list()

    def get(self):
        return f"accuracy: {sum(self.values)/len(self.values) * 100}%"