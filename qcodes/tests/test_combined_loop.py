from unittest import TestCase
import numpy as np
from hypothesis import given, settings
import hypothesis.strategies as hst


from .instrument_mocks import DummyInstrument
from qcodes.instrument.parameter import combine
from qcodes import Task, Loop
from qcodes.instrument.parameter import Parameter


class TestLoopCombined(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dmm = DummyInstrument(name="dmm", gates=['voltage', 'somethingelse'])
        cls.dmm.somethingelse.get = lambda: 1

    @classmethod
    def tearDownClass(cls):
        cls.dmm.close()
        del cls.dmm

    @given(npoints=hst.integers(2, 100),
           x_start_stop=hst.lists(hst.integers(min_value=-800, max_value=400),
                                  min_size=2, max_size=2, unique=True).map(sorted),
           y_start_stop=hst.lists(hst.integers(min_value=-800, max_value=400),
                                  min_size=2, max_size=2, unique=True).map(sorted),
           z_start_stop=hst.lists(hst.integers(min_value=-800, max_value=400),
                                  min_size=2, max_size=2, unique=True).map(sorted))
    @settings(max_examples=10, deadline=300)
    def testLoopCombinedParameterPrintTask(self, npoints, x_start_stop, y_start_stop, z_start_stop):

        x_set = np.linspace(x_start_stop[0], x_start_stop[1], npoints)
        y_set = np.linspace(y_start_stop[0], y_start_stop[1], npoints)
        z_set = np.linspace(z_start_stop[0], z_start_stop[1], npoints)
        setpoints = np.hstack((x_set.reshape(npoints, 1),
                               y_set.reshape(npoints, 1),
                               z_set.reshape(npoints, 1)))

        parameters = [Parameter(name, get_cmd=None, set_cmd=None) for name in ["X", "Y", "Z"]]

        sweep_values = combine(*parameters,
                               name="combined").sweep(setpoints)
        def ataskfunc():
            a = 1+1

        def btaskfunc():
            b = 1+2

        atask = Task(ataskfunc)
        btask = Task(btaskfunc)

        loop = Loop(sweep_values).each(atask, btask)
        data = loop.run(quiet=True)
        np.testing.assert_array_equal(data.arrays['X'].ndarray, x_set)
        np.testing.assert_array_equal(data.arrays['Y'].ndarray, y_set)
        np.testing.assert_array_equal(data.arrays['Z'].ndarray, z_set)

    @given(npoints=hst.integers(2, 100),
           x_start_stop=hst.lists(hst.integers(min_value=-800, max_value=400),
                                  min_size=2, max_size=2, unique=True).map(sorted),
           y_start_stop=hst.lists(hst.integers(min_value=-800, max_value=400),
                                  min_size=2, max_size=2, unique=True).map(sorted),
           z_start_stop=hst.lists(hst.integers(min_value=-800, max_value=400),
                                  min_size=2, max_size=2, unique=True).map(sorted))
    @settings(max_examples=10, deadline=300)
    def testLoopCombinedParameterTwice(self, npoints, x_start_stop, y_start_stop, z_start_stop):
        x_set = np.linspace(x_start_stop[0], x_start_stop[1], npoints)
        y_set = np.linspace(y_start_stop[0], y_start_stop[1], npoints)
        z_set = np.linspace(z_start_stop[0], z_start_stop[1], npoints)
        setpoints = np.hstack((x_set.reshape(npoints, 1),
                               y_set.reshape(npoints, 1),
                               z_set.reshape(npoints, 1)))
        parameters = [Parameter(name, get_cmd=None, set_cmd=None) for name in ["X", "Y", "Z"]]
        sweep_values = combine(*parameters,
                               name="combined").sweep(setpoints)

        def wrapper():
            counter = 0

            def inner():
                nonlocal counter
                counter += 1
                return counter

            return inner

        self.dmm.voltage.get = wrapper()
        loop = Loop(sweep_values).each(self.dmm.voltage, self.dmm.voltage)
        data = loop.run(quiet=True)
        np.testing.assert_array_equal(data.arrays['X'].ndarray, x_set)
        np.testing.assert_array_equal(data.arrays['Y'].ndarray, y_set)
        np.testing.assert_array_equal(data.arrays['Z'].ndarray, z_set)
        np.testing.assert_array_equal(data.arrays['dmm_voltage_0'].ndarray, np.arange(1, npoints*2, 2))
        np.testing.assert_array_equal(data.arrays['dmm_voltage_1'].ndarray, np.arange(2, npoints*2+1, 2))

    @given(npoints=hst.integers(2, 100),
           x_start_stop=hst.lists(hst.integers(min_value=-800, max_value=400),
                                  min_size=2, max_size=2, unique=True).map(sorted),
           y_start_stop=hst.lists(hst.integers(min_value=-800, max_value=400),
                                  min_size=2, max_size=2, unique=True).map(sorted),
           z_start_stop=hst.lists(hst.integers(min_value=-800, max_value=400),
                                  min_size=2, max_size=2, unique=True).map(sorted))
    @settings(max_examples=10, deadline=600)
    def testLoopCombinedParameterAndMore(self, npoints, x_start_stop, y_start_stop, z_start_stop):
        x_set = np.linspace(x_start_stop[0], x_start_stop[1], npoints)
        y_set = np.linspace(y_start_stop[0], y_start_stop[1], npoints)
        z_set = np.linspace(z_start_stop[0], z_start_stop[1], npoints)
        setpoints = np.hstack((x_set.reshape(npoints, 1),
                               y_set.reshape(npoints, 1),
                               z_set.reshape(npoints, 1)))
        parameters = [Parameter(name, get_cmd=None, set_cmd=None) for name in ["X", "Y", "Z"]]
        sweep_values = combine(*parameters,
                               name="combined").sweep(setpoints)

        def wrapper():
            counter = 0

            def inner():
                nonlocal counter
                counter += 1
                return counter

            return inner

        self.dmm.voltage.get = wrapper()
        loop = Loop(sweep_values).each(self.dmm.voltage, self.dmm.somethingelse, self.dmm.voltage)
        data = loop.run(quiet=True)
        np.testing.assert_array_equal(data.arrays['X'].ndarray, x_set)
        np.testing.assert_array_equal(data.arrays['Y'].ndarray, y_set)
        np.testing.assert_array_equal(data.arrays['Z'].ndarray, z_set)
        np.testing.assert_array_equal(data.arrays['dmm_voltage_0'].ndarray, np.arange(1, npoints * 2, 2))
        np.testing.assert_array_equal(data.arrays['dmm_somethingelse'].ndarray, np.ones(npoints))
        np.testing.assert_array_equal(data.arrays['dmm_voltage_2'].ndarray, np.arange(2, npoints * 2 + 1, 2))

    @given(npoints=hst.integers(2, 50),
           npoints_outer=hst.integers(2,25),
           x_start_stop=hst.lists(hst.integers(min_value=-800, max_value=400),
                                  min_size=2, max_size=2, unique=True).map(sorted),
           y_start_stop=hst.lists(hst.integers(min_value=-800, max_value=400),
                                  min_size=2, max_size=2, unique=True).map(sorted),
           z_start_stop=hst.lists(hst.integers(min_value=-800, max_value=400),
                                  min_size=2, max_size=2, unique=True).map(sorted))
    @settings(max_examples=10, deadline=None)
    def testLoopCombinedParameterInside(self, npoints, npoints_outer, x_start_stop, y_start_stop, z_start_stop):
        x_set = np.linspace(x_start_stop[0], x_start_stop[1], npoints_outer)
        y_set = np.linspace(y_start_stop[0], y_start_stop[1], npoints)
        z_set = np.linspace(z_start_stop[0], z_start_stop[1], npoints)

        setpoints = np.hstack((y_set.reshape(npoints, 1),
                               z_set.reshape(npoints, 1)))

        parameters = [Parameter(name, get_cmd=None, set_cmd=None) for name in ["X", "Y", "Z"]]
        sweep_values = combine(parameters[1], parameters[2],
                               name="combined").sweep(setpoints)

        def ataskfunc():
            a = 1+1

        def btaskfunc():
            b = 1+2

        atask = Task(ataskfunc)
        btask = Task(btaskfunc)


        def wrapper():
            counter = 0

            def inner():
                nonlocal counter
                counter += 1
                return counter

            return inner

        self.dmm.voltage.get = wrapper()
        loop = Loop(parameters[0].sweep(x_start_stop[0],
                                     x_start_stop[1],
                                     num=npoints_outer)).loop(sweep_values).each(self.dmm.voltage,
                                                                                 atask,
                                                                                 self.dmm.somethingelse,
                                                                                 self.dmm.voltage,
                                                                                 btask)
        data = loop.run(quiet=True)
        np.testing.assert_array_equal(data.arrays['X_set'].ndarray, x_set)
        np.testing.assert_array_equal(data.arrays['Y'].ndarray,
                                      np.repeat(y_set.reshape(1,npoints), npoints_outer, axis=0))
        np.testing.assert_array_equal(data.arrays['Z'].ndarray,
                                      np.repeat(z_set.reshape(1, npoints), npoints_outer, axis=0))

        np.testing.assert_array_equal(data.arrays['dmm_voltage_0'].ndarray,
                                      np.arange(1, npoints * npoints_outer* 2, 2).reshape(npoints_outer, npoints))
        np.testing.assert_array_equal(data.arrays['dmm_voltage_3'].ndarray,
                                      np.arange(2, npoints * npoints_outer* 2 + 1, 2).reshape(npoints_outer, npoints))
        np.testing.assert_array_equal(data.arrays['dmm_somethingelse'].ndarray, np.ones((npoints_outer, npoints)))
