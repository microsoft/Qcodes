"""
This module is intended to provide various methods to monitor in real time
data that is added to the data set
"""
import sys
import matplotlib.pyplot as plt
import numpy as np
import time


class DynamicPrint:
    """
    The dynamic print can be used as a subscriber which will refresh a single
    line on the stdout with the latest inserted data. To be robust against
    print statements called during the data insertion loop, the standard
    output is captured. The standard output is freed again if no data insertion
    has occurred within a refresh time. Once the standard output has been freed
    calling DynamicPrint again will raise a runtime error.
    """
    def __init__(self, names, refresh_rate):
        """
        Params:
        names (list): A list of strings in the format <title>:<format>
            For example: names=["x:.2e", "y:.3e"] which means that
            "x" will be printed with 2 significant digits with scientific
            notation.

        refresh_rate (float): How often the stdout line in refreshed. If
            inproperly chosen there is a risk that stdout will be released,
            which will cause a runtime error when the next call is made.
        """

        fmt = [self.split_format(name, ":") for name in names]
        self.print_template = ", ".join(["{} = {}".format(*f) for f in fmt])

        self.refresh_rate = refresh_rate

        self._real_std_out = sys.stdout
        self._buffer = {}
        self._last_write_time = None
        self._tcurrent = None
        self.has_captured = False
        self.has_released = False

    def write(self, strg):
        self._tcurrent = time.time()
        self._buffer[self._tcurrent] = strg

        if self._last_write_time is None or \
           self._tcurrent - self._last_write_time:

            self._write()
            self._last_write_time = self._tcurrent

    def _write(self):

        buffer_items = list(self._buffer.items())
        _, strg = buffer_items.pop(-1)

        while not strg.startswith("dynamic:") and len(buffer_items):
            t, _strg = buffer_items.pop(-1)

            if self._tcurrent - t > self.refresh_rate:
                break

            strg = _strg

        if not strg.startswith("dynamic:"):
            self.release_stdout()
        else:
            strg = strg.strip("dynamic:")

        self._real_std_out.write(strg)

    def flush(self):
        self._real_std_out.flush()

    def release_stdout(self):
        self.has_released = True
        sys.stdout = self._real_std_out

    def capture_stdout(self):
        sys.stdout = self
        self.has_captured = True

    @staticmethod
    def split_format(txt, splitter):
        if splitter not in txt:
            return txt, "{}"

        splits = txt.split(splitter)
        if len(splits) > 2:
            raise ValueError(f"'{txt}' has invalid format")

        splits[1] = "{:" + splits[1] + "}"
        return splits

    def __call__(self, results, length, state):

        if self.has_released:
            raise RuntimeError(
                "Live monitor disrupted. This can happen if the"
                "chosen refresh rate is improper"
            )

        if not self.has_captured:
            self.capture_stdout()

        s = self.print_template.format(*results[-1])
        self.write("dynamic:\r\x1b[K" + s.__str__())
        self.flush()


class Plotter:
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