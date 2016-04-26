from unittest import TestCase
import numpy as np

import qcodes 
import matplotlib
import matplotlib.pyplot as plt

class TestQtPlot(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_creation(self):
        ''' Simple test function which created a QtPlot window '''
        plotQ = qcodes.QtPlot(remote=False, show=False)


class TestMatPlot(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_creation(self):
        ''' Simple test function which created a QtPlot window '''
        plotM = qcodes.MatPlot()
        plt.close(plotM.fig)
