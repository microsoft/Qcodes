from unittest import TestCase
from unittest.mock import patch
import time
import multiprocessing as mp
import numpy as np

from qcodes.loops import Loop, MP_NAME, get_bg, halt_bg, Task, Wait, ActiveLoop
from qcodes.station import Station
from qcodes.data.io import DiskIO
from qcodes.data.data_array import DataArray
from qcodes.data.manager import get_data_manager
from qcodes.instrument.parameter import Parameter, ManualParameter
from qcodes.utils.multiprocessing import QcodesProcess
from qcodes.utils.validators import Numbers
from qcodes.utils.helpers import killprocesses, LogCapture
from .instrument_mocks import AMockModel, MockGates, MockSource, MockMeter


class TestMockInstLoop(TestCase):
    def setUp(self):
        get_data_manager().restart(force=True)
        killprocesses()
        # TODO: figure out what's leaving DataManager in a weird state
        # and fix it
        get_data_manager().restart(force=True)
        time.sleep(0.1)

        self.model = AMockModel()

        self.gates = MockGates(model=self.model)
        self.source = MockSource(model=self.model)
        self.meter = MockMeter(model=self.model)
        self.location = '_loop_test_'
        self.location2 = '_loop_test2_'
        self.io = DiskIO('.')

        c1 = self.gates.chan1
        self.loop = Loop(c1[1:5:1], 0.001).each(c1)

        self.assertFalse(self.io.list(self.location))
        self.assertFalse(self.io.list(self.location2))

    def tearDown(self):
        for instrument in [self.gates, self.source, self.meter]:
            instrument.close()

        get_data_manager().close()
        self.model.close()

        self.io.remove_all(self.location)
        self.io.remove_all(self.location2)

    def check_empty_data(self, data):
        expected = repr([float('nan')] * 4)
        self.assertEqual(repr(data.chan1.tolist()), expected)
        self.assertEqual(repr(data.chan1_set.tolist()), expected)

    def check_loop_data(self, data):
        self.assertEqual(data.chan1.tolist(), [1, 2, 3, 4])
        self.assertEqual(data.chan1_set.tolist(), [1, 2, 3, 4])

        self.assertTrue(self.io.list(self.location))

    def test_background_and_datamanager(self):
        # make sure that an unpicklable instrument can indeed run in a loop

        # TODO: if we don't save the dataset (location=False) then we can't
        # sync it when we're done. Should fix that - for now that just means
        # you can only do in-memory loops if you set data_manager=False
        # TODO: this is the one place we don't do quiet=True - test that we
        # really print stuff?
        data = self.loop.run(location=self.location)
        self.check_empty_data(data)

        # wait for process to finish (ensures that this was run in the bg,
        # because otherwise there *is* no loop.process)
        self.loop.process.join()

        data.sync()
        self.check_loop_data(data)

    def test_background_no_datamanager(self):
        data = self.loop.run(location=self.location, data_manager=False,
                             quiet=True)
        self.check_empty_data(data)

        self.loop.process.join()

        data.sync()
        self.check_loop_data(data)

    def test_foreground_and_datamanager(self):
        data = self.loop.run(location=self.location, background=False,
                             quiet=True)
        self.assertFalse(hasattr(self.loop, 'process'))

        self.check_loop_data(data)

    def test_foreground_no_datamanager(self):
        data = self.loop.run(location=self.location, background=False,
                             data_manager=False, quiet=True)
        self.assertFalse(hasattr(self.loop, 'process'))

        self.check_loop_data(data)

    def test_enqueue(self):
        c1 = self.gates.chan1
        loop = Loop(c1[1:5:1], 0.01).each(c1)
        loop.run(location=self.location, quiet=True)

        with self.assertRaises(RuntimeError):
            loop.run(location=self.location2, quiet=True)
        loop.run(location=self.location2, quiet=True, enqueue=True)
        loop.process.join()


def sleeper(t):
    time.sleep(t)


class TestBG(TestCase):
    def test_get_halt(self):
        killprocesses()
        self.assertIsNone(get_bg())

        p1 = QcodesProcess(name=MP_NAME, target=sleeper, args=(10, ))
        p1.start()
        p2 = QcodesProcess(name=MP_NAME, target=sleeper, args=(10, ))
        p2.start()
        p1.signal_queue = p2.signal_queue = mp.Queue()
        qcodes_processes = [p for p in mp.active_children()
                            if isinstance(p, QcodesProcess)]
        self.assertEqual(len(qcodes_processes), 2, mp.active_children())

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
        killprocesses()
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

    @patch('time.sleep')
    def test_delay0(self, sleep_mock):
        self.p2.set(3)

        loop = Loop(self.p1[1:3:1]).each(self.p2)

        self.assertEqual(loop.delay, 0)

        data = loop.run_temp()
        self.assertEqual(data.p1.tolist(), [1, 2])
        self.assertEqual(data.p2.tolist(), [3, 3])

        self.assertEqual(sleep_mock.call_count, 0)

    def test_bad_delay(self):
        for val, err in [(-1, ValueError), (-0.1, ValueError),
                         (None, TypeError), ('forever', TypeError)]:
            with self.assertRaises(err):
                Loop(self.p1[1:3:1], val)

            with self.assertRaises(err):
                Wait(val)

    def test_bare_wait(self):
        # Wait gets transformed to a Task, but is also callable on its own
        t0 = time.perf_counter()
        Wait(0.05)()
        delay = time.perf_counter() - t0
        # TODO: On Mac delay is always at least the time you waited, but on
        # Windows it is sometimes less? need to investigate the precision here.
        self.assertGreaterEqual(delay, 0.04)
        self.assertLessEqual(delay, 0.06)

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
        mg.setpoints = (None, (sp_dataarray,))

        data = loop.run_temp()
        self.assertEqual(data.bgn.tolist(), [[6, 7]] * 2)
        self.assertEqual(data.bgn.label, new_sp_label)

        # muck things up and test for errors
        mg.setpoints = (None, ((1, 2, 3),))
        with self.assertRaises(ValueError):
            loop.run_temp()

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

        # 2D size
        mg = MultiGetter(arr2d=((21, 22), (23, 24)))
        loop = Loop(self.p1[1:3:1], 0.001).each(mg)
        data = loop.run_temp()

        self.assertEqual(data.p1.tolist(), [1, 2])
        self.assertEqual(data.arr2d.tolist(), [[[21, 22], [23, 24]]] * 2)
        self.assertEqual(data.index0.tolist(), [[0, 1]] * 2)
        self.assertEqual(data.index1.tolist(), [[[0, 1]] * 2] * 2)

    def test_bad_actors(self):
        # would be nice to find errors at .each, but for now we find them
        # at .run
        loop = Loop(self.p1[1:3:1], 0.001).each(self.p1, 42)
        with self.assertRaises(TypeError):
            loop.run_temp()

        # at least invalid sweep values we find at .each
        with self.assertRaises(ValueError):
            Loop(self.p1[-20:20:1], 0.001).each(self.p1)

    def test_very_short_delay(self):
        with LogCapture() as s:
            Loop(self.p1[1:3:1], 1e-9).each(self.p1).run_temp()

        logstr = s.getvalue()
        s.close()
        self.assertEqual(logstr.count('negative delay'), 2, logstr)

    def test_zero_delay(self):
        with LogCapture() as s:
            Loop(self.p1[1:3:1]).each(self.p1).run_temp()

        logstr = s.getvalue()
        s.close()
        self.assertEqual(logstr.count('negative delay'), 0, logstr)


class AbortingGetter(ManualParameter):
    '''
    A manual parameter that can only be measured a couple of times
    before it aborts the loop that's measuring it.

    You have to attach the queue after construction with set_queue
    so you can grab it from the loop that uses the parameter.
    '''
    def __init__(self, *args, count=1, **kwargs):
        self._count = self._initial_count = count
        # also need a _signal_queue, but that has to be added later
        super().__init__(*args, **kwargs)

    def get(self):
        self._count -= 1
        if self._count <= 0:
            self._signal_queue.put(ActiveLoop.HALT)
        return super().get()

    def set_queue(self, queue):
        self._signal_queue = queue

    def reset(self):
        self._count = self._initial_count


class TestSignal(TestCase):
    def test_halt(self):
        p1 = AbortingGetter('p1', count=2, vals=Numbers(-10, 10))
        loop = Loop(p1[1:6:1], 0.005).each(p1)
        p1.set_queue(loop.signal_queue)

        with self.assertRaises(KeyboardInterrupt):
            loop.run_temp()

        data = loop.data_set

        nan = float('nan')
        self.assertEqual(data.p1.tolist()[:2], [1, 2])
        # when NaN is involved, I'll just compare reprs, because NaN!=NaN
        self.assertEqual(repr(data.p1.tolist()[-2:]), repr([nan, nan]))
        # because of the way the waits work out, we can get an extra
        # point measured before the interrupt is registered. But the
        # test would be valid either way.
        self.assertIn(repr(data.p1[2]), (repr(nan), repr(3), repr(3.0)))
