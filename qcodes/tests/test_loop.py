from unittest import TestCase
import time
import multiprocessing as mp
import numpy as np

from qcodes.loops import Loop, MP_NAME, get_bg, halt_bg, Task, Wait
from qcodes.station import Station
from qcodes.data.io import DiskIO
from qcodes.data.data_array import DataArray
from qcodes.instrument.parameter import Parameter, ManualParameter
from qcodes.utils.multiprocessing import QcodesProcess
from qcodes.utils.validators import Numbers
from .instrument_mocks import AMockModel, MockGates, MockSource, MockMeter


class TestMockInstLoop(TestCase):
    def setUp(self):
        self.model = AMockModel()

        self.gates = MockGates(self.model)
        self.source = MockSource(self.model)
        self.meter = MockMeter(self.model)
        self.location = '_loop_test_'
        self.io = DiskIO('.')

    def tearDown(self):
        self.io.remove_all(self.location)

    def test_instruments_in_loop(self):
        # make sure that an unpicklable instrument can indeed run in a loop
        self.assertFalse(self.io.list(self.location))
        c1 = self.gates.chan1
        loop = Loop(c1[1:5:1], 0.001).each(c1)

        # TODO: if we don't save the dataset (location=False) then we can't
        # sync it when we're done. Should fix that - for now that just means
        # you can only do in-memory loops if you set data_manager=False
        data = loop.run(location=self.location, quiet=True)

        # wait for process to finish (ensures that this was run in the bg,
        # because otherwise there *is* no loop.process)
        loop.process.join()

        data.sync()

        self.assertEqual(data.chan1.tolist(), [1, 2, 3, 4])
        self.assertEqual(data.chan1_set.tolist(), [1, 2, 3, 4])

        self.assertTrue(self.io.list(self.location))


def sleeper(t):
    time.sleep(t)


class TestGetHaltBG(TestCase):
    def test_get_halt(self):
        self.assertIsNone(get_bg())

        p1 = QcodesProcess(name=MP_NAME, target=sleeper, args=(10, ))
        p1.start()
        p2 = QcodesProcess(name=MP_NAME, target=sleeper, args=(10, ))
        p2.start()
        p1.signal_queue = p2.signal_queue = mp.Queue()
        self.assertEqual(len(mp.active_children()), 2)

        with self.assertRaises(RuntimeError):
            get_bg()
        bg1 = get_bg(return_first=True)
        self.assertIn(bg1, [p1, p2])

        halt_bg(timeout=0.01)
        bg2 = get_bg()
        self.assertIn(bg2, [p1, p2])
        # is this robust? requires that active_children always returns the same
        # order, even if it's not the order you started processes in
        self.assertNotEqual(bg1, bg2)

        self.assertEqual(len(mp.active_children()), 1)

        halt_bg(timeout=0.01)
        self.assertIsNone(get_bg())

        self.assertEqual(len(mp.active_children()), 0)

        # TODO - test that we print "no loops running"?
        # at least this shows that it won't raise an error
        halt_bg()


class FakeMonitor:
    '''
    when attached to an ActiveLoop as _monitor, records how long
    the monitor was given to measure
    '''
    def __init__(self, delay_array):
        self.delay_array = delay_array

    def call(self, finish_by=None):
        self.delay_array.append(finish_by - time.perf_counter())


class MultiGetter(Parameter):
    def __init__(self, **kwargs):
        if len(kwargs) == 1:
            name, self._return = list(kwargs.items())[0]
            super().__init__(name=name)
            self.size = np.shape(self._return)
        else:
            names = tuple(sorted(kwargs.keys()))
            super().__init__(names=names)
            self._return = tuple(kwargs[k] for k in names)
            self.sizes = tuple(np.shape(v) for v in self._return)

    def get(self):
        return self._return


class TestLoop(TestCase):
    def setUp(self):
        self.p1 = ManualParameter('p1', vals=Numbers(-10, 10))
        self.p2 = ManualParameter('p2', vals=Numbers(-10, 10))
        self.p3 = ManualParameter('p3', vals=Numbers(-10, 10))

    def test_nesting(self):
        loop = Loop(self.p1[1:3:1], 0.001).loop(
            self.p2[3:5:1], 0.001).loop(
            self.p3[5:7:1], 0.001)
        active_loop = loop.each(self.p1, self.p2, self.p3)
        data = active_loop.run_temp()

        self.assertEqual(data.p1_set.tolist(), [1, 2])
        self.assertEqual(data.p2_set.tolist(), [[3, 4]] * 2)
        self.assertEqual(data.p3_set.tolist(), [[[5, 6]] * 2] * 2)

        self.assertEqual(data.p1.tolist(), [[[1, 1]] * 2, [[2, 2]] * 2])
        self.assertEqual(data.p2.tolist(), [[[3, 3], [4, 4]]] * 2)
        self.assertEqual(data.p3.tolist(), [[[5, 6]] * 2] * 2)

    def test_default_measurement(self):
        Station().set_measurement(self.p2, self.p3)

        self.p2.set(4)
        self.p3.set(5)

        data = Loop(self.p1[1:3:1], 0.001).run_temp()

        self.assertEqual(data.p1.tolist(), [1, 2])
        self.assertEqual(data.p2.tolist(), [4, 4])
        self.assertEqual(data.p3.tolist(), [5, 5])

        data = Loop(self.p1[1:3:1], 0.001).each(
            Loop(self.p2[3:5:1], 0.001)).run_temp()

        self.assertEqual(data.p1.tolist(), [1, 2])
        self.assertEqual(data.p2.tolist(), [[3, 4], [3, 4]])
        self.assertEqual(data.p2_set.tolist(), [[3, 4], [3, 4]])
        self.assertEqual(data.p3.tolist(), [[5, 5]] * 2)

    def test_tasks_waits(self):
        delay0 = 0.01
        delay1 = 0.03
        loop = Loop(self.p1[1:3:1], delay0).each(
            Task(self.p2.set, -1),
            Wait(delay1),
            self.p2,
            Task(self.p2.set, 1),
            self.p2)
        delay_array = []
        loop._monitor = FakeMonitor(delay_array)

        data = loop.run_temp()

        self.assertEqual(data.p1.tolist(), [1, 2])
        self.assertEqual(data.p2_2.tolist(), [-1, -1])
        self.assertEqual(data.p2_4.tolist(), [1, 1])

        self.assertEqual(len(delay_array), 4)
        for i, delay in enumerate(delay_array):
            target = delay1 if i % 2 else delay0
            self.assertLessEqual(delay, target)
            self.assertGreater(delay, target - 0.001)

    def test_composite_params(self):
        # this one has names and sizes
        mg = MultiGetter(one=1, onetwo=(1, 2))
        self.assertTrue(hasattr(mg, 'names'))
        self.assertTrue(hasattr(mg, 'sizes'))
        self.assertFalse(hasattr(mg, 'name'))
        self.assertFalse(hasattr(mg, 'size'))
        loop = Loop(self.p1[1:3:1], 0.001).each(mg)
        data = loop.run_temp()

        self.assertEqual(data.p1.tolist(), [1, 2])
        self.assertEqual(data.one.tolist(), [1, 1])
        self.assertEqual(data.onetwo.tolist(), [[1, 2]] * 2)
        self.assertEqual(data.index0.tolist(), [[0, 1]] * 2)

        # give it setpoints, names, and labels
        mg.setpoints = (None, ((10, 11),))
        sp_name = 'highest'
        mg.setpoint_names = (None, (sp_name,))
        sp_label = 'does it go to 11?'
        mg.setpoint_labels = (None, (sp_label,))

        data = loop.run_temp()

        self.assertEqual(data.highest.tolist(), [[10, 11]] * 2)
        self.assertEqual(data.highest.label, sp_label)

        # setpoints as DataArray - name and label here override
        # setpoint_names and setpoint_labels attributes
        new_sp_name = 'bgn'
        new_sp_label = 'boogie nights!'
        sp_dataarray = DataArray(preset_data=[6, 7], name=new_sp_name,
                                 label=new_sp_label)
        # del mg.setpoint_names
        # sp_dataarray.name = new_sp_name
        # del mg.setpoint_labels
        # sp_dataarray.label = new_sp_label
        mg.setpoints = (None, (sp_dataarray,))

        data = loop.run_temp()
        self.assertEqual(data.bgn.tolist(), [[6, 7]] * 2)
        self.assertEqual(data.bgn.label, new_sp_label)

        # muck things up and test for errors
        mg.setpoints = (None, ((1, 2), (3, 4)))
        with self.assertRaises(ValueError):
            loop.run_temp()
        del mg.setpoints, mg.setpoint_names, mg.setpoint_labels
        mg.names = mg.names + ('extra',)
        with self.assertRaises(ValueError):
            loop.run_temp()
        del mg.names
        with self.assertRaises(ValueError):
            loop.run_temp()

        # this one has name and size
        mg = MultiGetter(arr=(4, 5, 6))
        self.assertTrue(hasattr(mg, 'name'))
        self.assertTrue(hasattr(mg, 'size'))
        self.assertFalse(hasattr(mg, 'names'))
        self.assertFalse(hasattr(mg, 'sizes'))
        loop = Loop(self.p1[1:3:1], 0.001).each(mg)
        data = loop.run_temp()

        self.assertEqual(data.p1.tolist(), [1, 2])
        self.assertEqual(data.arr.tolist(), [[4, 5, 6]] * 2)
        self.assertEqual(data.index0.tolist(), [[0, 1, 2]] * 2)

        # alternate form for 1D size, just an integer
        mg.size = mg.size[0]
        loop = Loop(self.p1[1:3:1], 0.001).each(mg)
        data = loop.run_temp()

        self.assertEqual(data.p1.tolist(), [1, 2])
        self.assertEqual(data.arr.tolist(), [[4, 5, 6]] * 2)
