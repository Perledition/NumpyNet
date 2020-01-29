# import standard modules
import random as rm
import datetime as dt

# import third party modules
import numpy as np


class Sequential:

    def __init__(self, layers, epochs=100, batch_size=100):

        self.layers = layers  # list of layers to build a network from folder layers
        self.epochs = epochs  # number of epochs to train
        self.batch_size = batch_size  # size of one batch for processing
        self.loss = list()  # empty list to store loss values
        self.acc = list()  # empty list to store accuracy values

    def calculate_cost(self, prediction, y):
        # SUPPORTS ONLY CATEGORICAL CROSS ENTROPY SO FAR SINCE MEANT TO BE A SIMPLE CNN FRAMEWORK
        return -1/self.batch_size * np.dot(y, np.log(prediction).T)

    @staticmethod
    def status_update(cost, batch_nr, batch_size, epoch, time, acc):
        if batch_nr == batch_size - 1:
            print(f'epoch {epoch}: {"=" * 100 + ">"}|{(batch_nr / batch_size):.5f} | cost: {cost:.5f} | acc: {acc:.2f}% | duration: {time}')
        else:
            print(
                f'epoch {epoch}: {"=" * (100 - int((1 - ((batch_nr + 1) / batch_size)) * 100)) + ">" + " " * int((1 - ((batch_nr + 1) / batch_size)) * 100)}|{((batch_nr + 1) / batch_size):.5f}% | cost: {cost:.5f} | acc: {acc:.2f}% | duration: {time}')

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
    def accuracy(x, y):
        idx_x = np.argmax(x, axis=1)
        idx_y = np.argmax(y.transpose(), axis=1)
        count = 0
        for i in range(idx_x.shape[0]):
            if idx_x[i] == idx_y[i]:
                count += 1
        return count/idx_y.shape[0] * 100

    def batch_processing(self, x, y):

        samples = x
        for nr, x_return in enumerate(samples):
            # forward propagation for all layers
            results = list()
            for i, layer in enumerate(self.layers):
                x_return = layer.forward(x_return)
                results.append(x_return.copy())

            acc = 1 if np.argmax(x_return, axis=1) == y else 0
            cost = self.calculate_cost(x_return, y[:, nr].transpose())

            # backward propagation for all layers
            dcost = cost.copy()
            for i in range(len(self.layers)):
                print('backward layer: ', -i)
                dcost = self.layers[-i].backward(dcost)

        return cost, acc

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
                cost, acc = self.batch_processing(tx, ty)
                self.loss.append(cost)
                self.acc.append(acc)

                # status update
                duration = str(dt.datetime.now() - batch_start)
                if (ix is 0) or (len(batches) % ix is 0) or (ix == len(batches) - 1):
                    self.status_update(cost, ix, len(batches), epoch, duration, acc)

            print(f'epoch {epoch}: {dt.datetime.now() - epoch_start}\n  ')
            print(f"<---------------- EPOCH: {epoch + 1} ------------------->")
    

