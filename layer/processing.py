# import standard modules
import random as rm
import datetime as dt

# import third party modules
import numpy as np


class Sequential:

    def __init__(self, layers, epochs=20, batch_size=100):
        self.layers = layers
        self.epochs = epochs
        self.batch_size = batch_size

    @staticmethod
    def status_update(cost, batch_nr, batch_size, epoch, time):
        if batch_nr == batch_size - 1:
            print(f'epoch {epoch}: {"=" * 100 + ">"}|{batch_nr / batch_size} | cost: {cost} | duration: {time}')
        else:
            print(
                f'epoch {epoch}: {"=" * (100 - int((1 - ((batch_nr + 1) / batch_size)) * 100)) + ">" + " " * int((1 - ((batch_nr + 1) / batch_size)) * 100)}|{(batch_nr + 1) / batch_size}% | cost: {cost} | duration: {time}')

    def batch_generator(self, x, y):
        # shuffle x and y together to make sure that all batches contain more than one class
        c = list(zip(x, list(y.transpose())))
        rm.shuffle(c)

        # initialize starting parameters for creating batch sizes
        batches = list()
        batches_count = int(len(x) / self.batch_size)
        start = 0
        end = self.batch_size

        # start splitting samples into batches
        for i in range(1, batches_count + 1):
            if i + 1 < batches_count:
                batches.append(c[start:end])
            else:
                batches.append(c[start:])

            start += self.batch_size
            end += self.batch_size

        return batches

    @staticmethod
    def calculate_cost(yh, y):
        return -1/yh.shape[0] * np.dot(y.reshape(-1), np.log2(yh.reshape(-1)).T)

    def batch_processing(self, x, y):

        x_return = x
        # forward propagation for all layers
        results = list()
        for i, layer in enumerate(self.layers):
            x_return = layer.assign(x_return)
            results.append(x_return.copy())

        cost = self.calculate_cost(x_return, y)

        # backward propagation for all layers
        for i in range(len(self.layers)):
            try:
                x_return = self.layers[-i].backward_propagation(x_return, results[-i])
            except Exception as e:
                # print(e)
                continue

        return cost

    def train(self, x, y):
        print('create batches for training process...')
        batches = self.batch_generator(x, y)
        print(f'{len(batches)} were created with samples size of {self.batch_size} each..')
        print('start training...')

        for epoch in range(self.epochs):
            epoch_start = dt.datetime.now()
            for ix, batch in enumerate(batches):
                batch_start = dt.datetime.now()
                # decode batch into usable x and y since it is through the batch generator in a list
                tx, ty = zip(*batch)
                ty = np.array(ty).transpose()
                # print('x in:', tx)
                # print('x in:', len(tx), tx[0].shape)

                cost = self.batch_processing(tx, ty)

                # status update
                duration = str(dt.datetime.now() - batch_start)
                self.status_update(cost, ix, len(batches), epoch, duration)

            print(f'epoch {epoch}: {dt.datetime.now() - epoch_start}')
