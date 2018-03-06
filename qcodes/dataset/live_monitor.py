"""
This module is intended to provide various methods to monitor in real time
data that is added to the data set
"""
import matplotlib.pyplot as plt
import numpy as np


class Plot1DSubscriber:
    """
    This class can be used to plot 1D data inserted in the data set. Plots will
    be updated in real time.

    Args
    ----
    axes_dict (dictionary): The keys represent axis labels and the values
        the indices in the results list to be plotted.
    """
    def __init__(self, axes_dict):
        self.x_index, self.y_index = axes_dict.values()
        self.x_name, self.y_name = axes_dict.keys()

        self._state = {"x": [], "y": []}

        self.fig, self.ax = plt.subplots(1)
        self.ax.set_xlabel(self.x_name)
        self.ax.set_ylabel(self.y_name)

        self.line, = self.ax.plot([], [])
        self.fig.show()
        plt.ion()

    @staticmethod
    def _calculate_bounding_box(state):
        x = state["x"]
        y = state["y"]
        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)

        dx = (xmax - xmin) * 1.1
        dy = (ymax - ymin) * 1.1
        cx, cy = (xmax + xmin) / 2, (ymax + ymin) / 2

        xmin = cx - dx / 2
        xmax = cx + dx / 2

        ymin = cy - dy / 2
        ymax = cy + dy / 2

        return {"x": (xmin, xmax), "y": (ymin, ymax)}

    def __call__(self, results, length, state=None):
        r = list(map(list, zip(*results)))
        xnew, ynew = [r[i] for i in [self.x_index, self.y_index]]

        self._state["x"] = np.append(self._state["x"], xnew)
        self._state["y"] = np.append(self._state["y"], ynew)

        if len(self._state["x"]) == 1:
            return

        self.line.set_xdata(self._state["x"])
        self.line.set_ydata(self._state["y"])
        bb = self._calculate_bounding_box(self._state)

        self.ax.set_xlim(*bb["x"])
        self.ax.set_ylim(*bb["y"])

        self.fig.canvas.draw()
