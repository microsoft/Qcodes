"""
Tests for plotting system.
Legacy in many ways:

    - assume X server running
    - just test "window creation"
"""
from unittest import TestCase, skipIf
import numpy as np
import os

try:
    from qcodes.plots.pyqtgraph import QtPlot
    if os.environ.get("TRAVISCI"):
        noQtPlot = True
    else:
        noQtPlot = False
except Exception:
    noQtPlot = True

try:
    from qcodes.plots.qcmatplotlib import MatPlot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if os.environ.get("TRAVISCI"):
        noMatPlot = True
    else:
        noMatPlot = False
except Exception:
    noMatPlot = True


@skipIf(noQtPlot, '***pyqtgraph plotting cannot be tested***')
class TestQtPlot(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_creation(self):
        """
        Simple test function which created a QtPlot window
        """
        plotQ = QtPlot(remote=False, show_window=False, interval=0)
        plotQ.add_subplot()

    def test_simple_plot(self):
        main_QtPlot = QtPlot(
            window_title='Main plotmon of TestQtPlot',
            figsize=(600, 400))

        x = np.arange(0, 10e-6, 1e-9)
        f = 2e6
        y = np.cos(2*np.pi*f*x)

        for j in range(4):
            main_QtPlot.add(x=x, y=y,
                            xlabel='Time', xunit='s',
                            ylabel='Amplitude', yunit='V',
                            subplot=j+1,
                            symbol='o', symbolSize=5)

    def test_return_handle(self):
        plotQ = QtPlot(remote=False)
        return_handle = plotQ.add([1, 2, 3])
        self.assertIs(return_handle, plotQ.subplots[0].items[0])


@skipIf(noMatPlot, '***matplotlib plotting cannot be tested***')
class TestMatPlot(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_creation(self):
        """
        Simple test function which created a MatPlot window
        """
        plotM = MatPlot(interval=0)
        plt.close(plotM.fig)

    def test_return_handle(self):
        plotM = MatPlot(interval=0)
        returned_handle = plotM.add([1, 2, 3])
        line_handle = plotM[0].get_lines()[0]
        self.assertIs(returned_handle, line_handle)
        plotM.clear()
        plt.close(plotM.fig)
