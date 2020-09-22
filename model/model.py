# import standard modules

# import third party modules
import numpy as np
from matplotlib import pyplot as plt

# import project related modules
from metric.measure import Accuracy
from loss.cost import RootMeanSquaredError, CrossEntropy


class Sequence(object):

    def __init__(self, layer: list, loss: str, epochs: int, batch_size: int):
        self.layer = layer
        self.loss = {"rmse": RootMeanSquaredError, "cross_entropy": CrossEntropy}[loss]()
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_history = list()
        self.performance = {"accuracy": Accuracy}["accuracy"]()

    def add(self, layer):
        self.layer.append(layer)

    def monitor_callback(self, X, Y, iteration):
        xx, yy = np.mgrid[X.min():X.max():.1, X.min():X.max():.1]
        grid = np.c_[xx.ravel(), yy.ravel()]
        preds = np.array([self.predict(grid[i]) for i in range(0, grid.shape[0])])
        preds = preds[:, 0, 0].reshape(xx.shape)

        y = Y[:, 0]

        f, ax = plt.subplots(figsize=(10, 10))
        contour = ax.contourf(xx, yy, preds, 100, cmap="Blues",
                              vmin=0, vmax=1)
        ax_c = f.colorbar(contour)
        ax_c.set_label("$P(y = 1)$")
        ax_c.set_ticks([0, .25, .5, .75, 1])

        ax.scatter(X[500:, 0], X[500:, 1], c=y[500:], s=50,
                   cmap="RdBu", vmin=-.2, vmax=1.2,
                   edgecolor="white", linewidth=1)

        ax.set(aspect="equal",
               xlim=(X.min(), X.max()), ylim=(X.min(), X.max()),
               xlabel="$X_1$", ylabel="$X_2$")

        save_path = os.path.join(os.getcwd(), 'numpynet_capture')
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        plt.savefig(os.path.join(save_path, f'{iteration}.png'))

    def predict(self, x):
        # iterate over each layer - forward feed
        for i, layer in enumerate(self.layer):

            # if it is the first iteration than handover sample reshaped
            if i is 0:
                z = self.layer[i].forward(x.reshape(-1, 1))

            # else hand over the current z value
            else:
                z = self.layer[i].forward(z)

        return z

    def calculated_batches(self, x):

        start = 0
        end = self.batch_size
        for i in range(0, int(x.shape[0]/self.batch_size)):
            batch = x[start:end, :]
            start += self.batch_size
            end += self.batch_size
            yield batch

    def _collect_loss(self, loss):
        self.loss_history.append(loss)

    def fit(self, x, y, capture_training=False):

        for e in range(1, self.epochs + 1):

            avg_loss = list()
            for ix in range(0, x.shape[0]):

                # forward feed
                z = x[ix].reshape(-1, 1)
                for i, layer in enumerate(self.layer):
                    z = self.layer[i].forward(z)

                # define cost
                self.performance.add(z, y[ix])
                cost = self.loss.forward(z, y[ix])
                avg_loss.append(cost)

                # define error
                error = self.loss.backward()

                # backward feed
                for index in range(1, len(self.layer) + 1):
                    error = self.layer[-index].backward(error)

            self.loss_history.append(sum(avg_loss)/len(avg_loss))

            if capture_training:
                self.monitor_callback(x, y, e)

            if e%50== 0:
                print(f"EPOCH {e}: avg. loss: {self.loss_history[-1]}, {self.performance.get()}")
            self.performance.clean()

    

