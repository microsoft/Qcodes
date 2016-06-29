from unittest import TestCase, skipIf

try:
    from qcodes.plots.pyqtgraph import QtPlot
    noQtPlot = False
except Exception:
    noQtPlot = True

try:
    from qcodes.plots.qcmatplotlib import MatPlot
    import matplotlib.pyplot as plt
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
        plotQ = QtPlot(remote=False, show_window=False, interval=0)
        _ = plotQ.add_subplot()


@skipIf(noMatPlot, '***matplotlib plotting cannot be tested***')
class TestMatPlot(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_creation(self):
        ''' Simple test function which created a QtPlot window '''
        plotM = MatPlot(interval=0)
        plt.close(plotM.fig)
