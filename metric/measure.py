# import standard modules

# import third party modules
import numpy as np

# import project related modules


class ClassificationMatrix:
    """
    creates a classification matrix and produces avg. precision and recall over all classes.
    the matrix is build as follows:
                        predicted class 1 | predicted class 2 | predicted n
        label class 1:          12        |        0          |      ...
        label class 1:          0         |        32          |      ...

    Therefore the label is produced as rows while the predicted output is on the column space

    """

    def __init__(self):
        self.matrix = None            # placeholder for the output matrix
        self.avg_precision = 0        # avg. precision resulting from the matrix
        self.avg_recall = 0           # avg. recall resulting from the matrix
        self.class_measures = dict()  # empty dict which will be filled with class specific recall and precision

    def _precision_and_recall(self, y: np.array):
        """
        class internal function to calculate precision and recall over all classes and a class specific
        recall and precision matrix which can be produced by calling the attribute class_measures directly

        :param y: numpy.array: label or target one hot encoded matrix
        :return: None
        """

        # set lists to collect recall and precision values over all classes
        recall_list = list()
        precision_list = list()

        # for each class calculate recall an precision form the matrix
        for target_class in range(y.shape[1]):

            # count true positive values
            tp = self.matrix[target_class, target_class]

            # count false negative values
            fn = np.sum(np.take(self.matrix[target_class],
                                [ix for ix in range(self.matrix.shape[0]) if ix != target_class]
                                )
                        )

            # count false positive values
            fp = np.sum(self.matrix[:, target_class]) - tp

            # calculate recall an precision for the class
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)

            # add class specific values to lists and create a class specific dict entry
            recall_list.append(recall)
            precision_list.append(precision)
            self.class_measures[target_class] = {"recall": recall, "precision": precision}

        # after iterating over all classes set avg precision and avg recall values
        self.avg_precision = sum(precision_list) / len(precision_list)
        self.avg_recall = sum(recall_list) / len(recall_list)

    def _calculate_matrix(self, y_hat: np.array, y: np.array):
        """
        class internal function which produces a classification matrix from a prediction and target matrix.
        both matrices must be one hot encoded.

        :param y_hat: numpy.array: of predictions
        :param y: numpy.array: of labels / targets
        :return: numpy.array: returns the produced classification matrix
        """

        # get the amount of unique classes and initialize and empty array in this quadratic dimension
        unique_classes = y.shape[1]
        self.matrix = np.zeros((unique_classes, unique_classes))

        # check for each sample the predictions and add plus one count
        for row in range(y.shape[0]):
            y_max = np.argmax(y[row, :])
            yh_max = np.argmax(y_hat[row, :])
            self.matrix[y_max, yh_max] += 1

        return self.matrix

    def fit(self, y_hat: np.array, y: np.array):
        """
        main function of the class to take predictions and labels in order to create classification matrix.
        y_hat and y should be one hot encoded matrices.

        :param y_hat: numpy.array: of predictions
        :param y: numpy.array: of labels / targets
        :return: numpy.array: returns the produced classification matrix
        """

        # ensure equal shape of target variables and predicted values
        assert y_hat.shape == y.shape, f"Shapes of predictions and targets do not fit {y_hat.shape} != {y.shape}"

        # first calculate the matrix from with the predictions and labels
        self.matrix = self._calculate_matrix(y_hat, y)

        # calculate precision and recall values
        self._precision_and_recall(y)

        # print avg. information. for more detailed information class_measures can be used.
        print(f"avg. precision: {round(self.avg_precision, 2)}%")
        print(f"avg. recall: {round(self.avg_recall, 2)}%")
        return self.matrix


class Accuracy:
    """
    class produces accuracy values and is able to collect values to return accuracy after a certain time or collection.
    This is needed because of the way the Sequence class of Numpy Net works - not the most elegant but effective way.
    """

    def __init__(self):
        self.values = list()   # list to collect all accuracy values over time

    def add(self, x: np.array, y: np.array):
        """
        add function calculates whether the prediction was successful or not from a single sample and adds a binary
        result to the global list.

        :param x: np.array: array with predicted results
        :param y: np.array: array with target label

        :return: None
        """

        # find the index of the max value for the given array
        pred = np.argmax(x)
        target = np.argmax(y)

        # if both arrays have the max value at the same position the result of a subtraction will be zero
        # and the prediction was successful. In this case a 1 is collected, otherwise a zero will be added to the list.
        if int(pred)-int(target) == 0:
            self.values.append(1)
        else:
            self.values.append(0)

    def clean(self):
        """
        cleans all values of the global list

        :return: None
        """
        self.values = list()

    def get(self):
        """
        provides the current avg of accuracy

        :return: str: string with value
        """

        return f"accuracy: {sum(self.values)/len(self.values) * 100}%"