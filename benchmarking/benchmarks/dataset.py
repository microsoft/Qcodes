"""
This module contains code used for benchmarking data saving speed of the
database used under the QCoDeS dataset.
"""
import shutil
import tempfile
import os

import numpy as np

import qcodes
from qcodes import ManualParameter
from qcodes.dataset.measurements import Measurement
from qcodes.dataset.experiment_container import new_experiment
from qcodes.dataset.database import initialise_database


class Adding5Params100Values200Times:
    def __init__(self):
        self.params = list()
        self.values = list()
        self.experiment = None
        self.runner = None
        self.datasaver = None
        self.tmpdir = None

    def setup(self):
        # Init DB
        self.tmpdir = tempfile.mkdtemp()
        qcodes.config["core"]["db_location"] = os.path.join(self.tmpdir, 'temp.db')
        qcodes.config["core"]["db_debug"] = False
        initialise_database()

        # Create experiment
        self.experiment = new_experiment("test-experiment", sample_name="test-sample")

        # Create measurement
        meas = Measurement(self.experiment)

        x1 = ManualParameter('x1')
        x2 = ManualParameter('x2')
        x3 = ManualParameter('x3')
        y1 = ManualParameter('y1')
        y2 = ManualParameter('y2')

        meas.register_parameter(x1)
        meas.register_parameter(x2)
        meas.register_parameter(x3)
        meas.register_parameter(y1, setpoints=[x1, x2, x3])
        meas.register_parameter(y2, setpoints=[x1, x2, x3])

        self.params = [x1, x2, x3, y1, y2]

        # Create the 'run' context manager
        self.runner = meas.run()

        # Enter Runner and create DataSaver
        self.datasaver = self.runner.__enter__()

        # Create values for parameters
        for _ in range(len(self.params)):
            self.values.append(np.random.rand(100))

    def teardown(self):
        # Exit runner context manager
        if self.runner:
            self.runner.__exit__(None, None, None)
            self.runner = None
            self.datasaver = None

        # Close DB connection
        if self.experiment:
            self.experiment.conn.close()
            self.experiment = None

        # Remove tmpdir with database
        if self.tmpdir:
            shutil.rmtree(self.tmpdir)
            self.tmpdir = None

        self.params = list()
        self.values = list()

    def time_test(self):
        """Adding data for 5 parameters, 100 values each, 200 times"""
        for _ in range(200):
            self.datasaver.add_result(
                (self.params[0], self.values[0]),
                (self.params[1], self.values[1]),
                (self.params[2], self.values[2]),
                (self.params[3], self.values[3]),
                (self.params[4], self.values[4])
            )


class Adding5Params10000Values2Times:
    def __init__(self):
        self.params = list()
        self.values = list()
        self.experiment = None
        self.runner = None
        self.datasaver = None
        self.tmpdir = None

    def setup(self):
        # Init DB
        self.tmpdir = tempfile.mkdtemp()
        qcodes.config["core"]["db_location"] = os.path.join(self.tmpdir, 'temp.db')
        qcodes.config["core"]["db_debug"] = False
        initialise_database()

        # Create experiment
        self.experiment = new_experiment("test-experiment", sample_name="test-sample")

        # Create measurement
        meas = Measurement(self.experiment)

        x1 = ManualParameter('x1')
        x2 = ManualParameter('x2')
        x3 = ManualParameter('x3')
        y1 = ManualParameter('y1')
        y2 = ManualParameter('y2')

        meas.register_parameter(x1)
        meas.register_parameter(x2)
        meas.register_parameter(x3)
        meas.register_parameter(y1, setpoints=[x1, x2, x3])
        meas.register_parameter(y2, setpoints=[x1, x2, x3])

        self.params = [x1, x2, x3, y1, y2]

        # Create the 'run' context manager
        self.runner = meas.run()

        # Enter Runner and create DataSaver
        self.datasaver = self.runner.__enter__()

        # Create values for parameters
        for _ in range(len(self.params)):
            self.values.append(np.random.rand(10000))

    def teardown(self):
        # Exit runner context manager
        if self.runner:
            self.runner.__exit__(None, None, None)
            self.runner = None
            self.datasaver = None

        # Close DB connection
        if self.experiment:
            self.experiment.conn.close()
            self.experiment = None

        # Remove tmpdir with database
        if self.tmpdir:
            shutil.rmtree(self.tmpdir)
            self.tmpdir = None

        self.params = list()
        self.values = list()

    def time_test(self):
        """Adding data for 5 parameters, 10000 values each, 2 times"""
        for _ in range(2):
            self.datasaver.add_result(
                (self.params[0], self.values[0]),
                (self.params[1], self.values[1]),
                (self.params[2], self.values[2]),
                (self.params[3], self.values[3]),
                (self.params[4], self.values[4])
            )
