from unittest import TestCase
import time
import multiprocessing as mp
import numpy as np

import qcodes as qc

class TestQtPlot(TestCase):
    def setUp(self):
	pass

    def tearDown(self):
	pass


    def test_creation(self):
        ''' Simple test function which created a QtPlot window '''
        plotQ = qc.QtPlot(remote=False, show=False)

