# import standard modules
import os

# import third party modules
import numpy as np
from matplotlib import pyplot as plt

# import project related modules
from layers.general import Layer
from metric.measure import Accuracy
from loss.cost import RootMeanSquaredError, CrossEntropy


# TODO: Classification Matrix, Precision and Recall
class Sequence(object):

    def __init__(self, layers: list, loss: str, epochs: int, feedback_steps: int = 50):
        self.layers = layers                      # list of layers the Sequence has to execute
        self.feedback_steps = feedback_steps      # optional value of after how many epochs a status statement appears
        self.epochs = epochs                      # defines how many epochs the algorithm runs
        self.loss_history = list()                # list in which the loss of each epoch is collected

        # performance measures used for the model or Sequence -
        self.performance = {"accuracy": Accuracy}["accuracy"]()

        # cost function used for the model or sequence - dict selection with keyword from parameter loss
        self.loss = {"rmse": RootMeanSquaredError, "cross_entropy": CrossEntropy}[loss]()

    def add(self, layer: Layer):
        """
        appends an additional Layer to the current architecture

        :param layer: Layer: must be Layer or child of a Layer class
        :return: None
        """
        # append new layer to list of layers
        self.layers.append(layer)

    def show(self):
        """
        function that print the current architecture of the Sequence
        :return: None
        """

        print("MODEL STRUCTURE:")
        main_divider = "========================================================="
        layer_divider = "---------------------------------------------------------"

        # create list to create a order
        print_list = list()
        print_list.append(main_divider)

        i = 0
        while i < len(self.layers):
            print_list.append(str(self.layers[i]))
            i += 1

            if i != len(self.layers):
                print_list.append(layer_divider)

        print(*print_list, sep="\n")

    def predict(self, x: np.array):
        """
        runs a prediction process for a given array of data.

        :param x: numpy.array: data points to predict can be one sample or a matrix of samples
        :return: numpy.array with values of predictions
        """

        # for each sample in input data run the prediction cycle
        predictions = list()
        for sample in range(x.shape[0]):

            # iterate over each layer - forward feed
            for i, layer in enumerate(self.layers):

                # if it is the first iteration than handover sample reshaped
                if i is 0:
                    z = self.layers[i].forward(x[sample].reshape(-1, 1))

                # else hand over the current z value
                else:
                    z = self.layers[i].forward(z)

            predictions.append(z[0])

        return np.array(predictions)

    def _collect_loss(self, loss: float):
        """
        class internal function to append a epoch loss value to the global loss history
        :param loss: float: value of loss to capture

        :return: None
        """

        self.loss_history.append(loss)

    def train(self, x: np.array, y: np.array):
        """
        training procedure of Sequence class. It's a sample by sample procedure so neither batch nor sgd are used for
        the training, this might be inefficient but is helpful in order to learn the mechanics and also simplifies the
        code quite a bit.

        :param x: numpy.array: training samples of feature data
        :param y: numpy.array: values target variables, must be one hot encoded for classification problem

        :return: None. After training the function predict can be used in order to make predictions with the model
        """

        # run the following loop for each epoch
        for e in range(1, self.epochs + 1):

            # initialize a list called average loss in order to collect the loss for each sample
            avg_loss = list()
            # for each sample in x run the forward and backward steps
            for ix in range(0, x.shape[0]):

                # forward feed push the sample of x through each layer in order to make predictions
                z = x[ix].reshape(-1, 1)
                for i, layer in enumerate(self.layers):
                    z = self.layers[i].forward(z)

                # get the current performance values form the performance function of initialization
                # and get the loss / cost of the current sample. the loss is append to the avg_loss list
                self.performance.add(z, y[ix])
                cost = self.loss.forward(z, y[ix])
                avg_loss.append(cost)

                # get the error between predicted value and target value
                error = self.loss.backward()

                # run the back propagation in order to adjust the weights
                for index in range(1, len(self.layers) + 1):
                    error = self.layers[-index].backward(error)

            # append the average loss after all samples went through the process above
            # the average loss is the avg. loss of the epoch
            self.loss_history.append(sum(avg_loss)/len(avg_loss))

            # if the current epoch is a feedback step size return information about the training.
            if e % self.feedback_steps == 0:
                print(f"EPOCH {e}: avg. loss: {self.loss_history[-1]}, {self.performance.get()}")

            # clean the performance values since for each epoch the measurement starts from zero
            self.performance.clean()

    

