from unittest import TestCase, skipIf
import matplotlib.pyplot as plt

try:
    from qcodes.plots.pyqtgraph import QtPlot
    noQtPlot = False
except Exception:
    noQtPlot = True

try:
    from qcodes.plots.matplotlib import MatPlot
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
        ''' Simple test function which created a QtPlot window '''
        plotQ = QtPlot(remote=False, interval=0)


@skipIf(noQtPlot, '***matplotlib plotting cannot be tested***')
class TestMatPlot(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_creation(self):
        ''' Simple test function which created a QtPlot window '''
        plotM = MatPlot(interval=0)
        plt.close(plotM.fig)
