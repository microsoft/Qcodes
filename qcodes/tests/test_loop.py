from datetime import datetime
import logging
import multiprocessing as mp
import numpy as np
import time
from unittest import TestCase
from unittest.mock import patch

from qcodes.loops import (Loop, MP_NAME, get_bg, halt_bg, ActiveLoop,
                          _DebugInterrupt)
from qcodes.actions import Task, Wait, BreakIf
from qcodes.station import Station
from qcodes.data.io import DiskIO
from qcodes.data.data_array import DataArray
from qcodes.data.manager import get_data_manager
from qcodes.instrument.parameter import Parameter, ManualParameter
from qcodes.process.helpers import kill_processes
from qcodes.process.qcodes_process import QcodesProcess
from qcodes.utils.validators import Numbers
from qcodes.utils.helpers import LogCapture

from .instrument_mocks import (AMockModel, MockGates, MockSource, MockMeter,
                               MultiGetter)


class TestMockInstLoop(TestCase):
    def setUp(self):
        get_data_manager().restart(force=True)
        kill_processes()
        # TODO: figure out what's leaving DataManager in a weird state
        # and fix it
        get_data_manager().restart(force=True)
        time.sleep(0.1)

        self.model = AMockModel()

        self.gates = MockGates(model=self.model, server_name='')
        self.source = MockSource(model=self.model, server_name='')
        self.meter = MockMeter(model=self.model, server_name='')
        self.location = '_loop_test_'
        self.location2 = '_loop_test2_'
        self.io = DiskIO('.')

        c1 = self.gates.chan1
        self.loop = Loop(c1[1:5:1], 0.001).each(c1)
        self.loop_progress = Loop(c1[1:5:1], 0.001,
                                  progress_interval=1).each(c1)

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
        self.assertEqual(repr(data.gates_chan1.tolist()), expected)
        self.assertEqual(repr(data.gates_chan1_set.tolist()), expected)

    def check_loop_data(self, data):
        self.assertEqual(data.gates_chan1.tolist(), [1, 2, 3, 4])
        self.assertEqual(data.gates_chan1_set.tolist(), [1, 2, 3, 4])

        self.assertTrue(self.io.list(self.location))

    def test_background_and_datamanager(self):
        # make sure that an unpicklable instrument can indeed run in a loop
        # because the instrument itself is in a server

        # TODO: if we don't save the dataset (location=False) then we can't
        # sync it when we're done. Should fix that - for now that just means
        # you can only do in-memory loops if you set data_manager=False
        # TODO: this is the one place we don't do quiet=True - test that we
        # really print stuff?
        data = self.loop.run(location=self.location, background=True)
        self.check_empty_data(data)

        # wait for process to finish (ensures that this was run in the bg,
        # because otherwise there *is* no loop.process)
        self.loop.process.join()

        data.sync()
        self.check_loop_data(data)

    def test_local_instrument(self):
        # a local instrument should work in a foreground loop, but
        # not in a background loop (should give a RuntimeError)
        self.gates.close()  # so we don't have two gates with same name
        gates_local = MockGates(model=self.model, server_name=None)
        self.gates = gates_local
        c1 = gates_local.chan1
        loop_local = Loop(c1[1:5:1], 0.001).each(c1)

        # if spawn, pickle will happen
        if mp.get_start_method() == "spawn":
            with self.assertRaises(RuntimeError):
                loop_local.run(location=self.location,
                               quiet=True,
                               background=True)
        # allow for *nix
        # TODO(giulioungaretti) see what happens ?
        # what is the expected beavhiour ?
        # The RunimError will never be raised here, as the forkmethod
        # won't try to pickle anything at all.
        else:
            logging.error("this should not be allowed, but for now we let it be")
            loop_local.run(location=self.location, quiet=True)

        data = loop_local.run(location=self.location2, background=False,
                              quiet=True)
        self.check_loop_data(data)

    def test_background_no_datamanager(self):
        data = self.loop.run(location=self.location,
                             background=True,
                             data_manager=False,
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

    def test_foreground_no_datamanager_progress(self):
        data = self.loop_progress.run(location=self.location, background=False,
                                      data_manager=False, quiet=True)
        self.assertFalse(hasattr(self.loop, 'process'))

        self.check_loop_data(data)

    @patch('qcodes.loops.tprint')
    def test_progress_calls(self, tprint_mock):
        data = self.loop_progress.run(location=self.location, background=False,
                                      data_manager=False, quiet=True)
        self.assertFalse(hasattr(self.loop, 'process'))

        self.check_loop_data(data)
        expected_calls = len(self.loop_progress.sweep_values) + 1
        self.assertEqual(tprint_mock.call_count, expected_calls)

        # now run again with no progress interval and check that we get no
        # additional calls
        data = self.loop_progress.run(location=False, background=False,
                                      data_manager=False, quiet=True,
                                      progress_interval=None)
        self.assertFalse(hasattr(self.loop, 'process'))

        self.check_loop_data(data)
        self.assertEqual(tprint_mock.call_count, expected_calls)

    def test_foreground_no_datamanager(self):
        data = self.loop.run(location=self.location, background=False,
                             data_manager=False, quiet=True)
        self.assertFalse(hasattr(self.loop, 'process'))

        self.check_loop_data(data)

    def test_enqueue(self):
        c1 = self.gates.chan1
        loop = Loop(c1[1:5:1], 0.01).each(c1)
        data1 = loop.run(location=self.location,
                         quiet=True,
                         background=True,
                         data_manager=True)

        # second running of the loop should be enqueued, blocks until
        # the first one finishes.
        # TODO: check what it prints?
        data2 = loop.run(location=self.location2,
                         quiet=True,
                         background=True,
                         data_manager=True)

        data1.sync()
        data2.sync()
        self.assertEqual(data1.gates_chan1.tolist(), [1, 2, 3, 4])
        for v in data2.gates_chan1:
            self.assertTrue(np.isnan(v))

        loop.process.join()
        data2.sync()
        self.assertEqual(data2.gates_chan1.tolist(), [1, 2, 3, 4])

        # and while we're here, check that running a loop in the
        # foreground *after* the background clears its .process
        self.assertTrue(hasattr(loop, 'process'))
        loop.run_temp()
        self.assertFalse(hasattr(loop, 'process'))


def sleeper(t):
    time.sleep(t)


class TestBG(TestCase):
    def test_get_halt(self):
        kill_processes()
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


class TestLoop(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p1 = ManualParameter('p1', vals=Numbers(-10, 10))
        cls.p2 = ManualParameter('p2', vals=Numbers(-10, 10))
        cls.p3 = ManualParameter('p3', vals=Numbers(-10, 10))
        Station().set_measurement(cls.p2, cls.p3)

    def setUp(self):
        kill_processes()

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

    def test_nesting_2(self):
        loop = Loop(self.p1[1:3:1]).each(
            self.p1,
            Loop(self.p2[3:5:1]).each(
                self.p1,
                self.p2,
                Loop(self.p3[5:7:1]).each(
                    self.p1,
                    self.p2,
                    self.p3)))

        data = loop.run_temp()
        keys = set(data.arrays.keys())

        self.assertEqual(data.p1_set.tolist(), [1, 2])
        self.assertEqual(data.p2_set.tolist(), [[3, 4]] * 2)
        self.assertEqual(data.p3_set.tolist(), [[[5, 6]] * 2] * 2)

        self.assertEqual(data.p1_0.tolist(), [1, 2])

        # TODO(alexcjohnson): these names are extra confusing...
        # perhaps we should say something like always include *all* indices
        # unless you can get rid of them all (ie that param only shows up
        # once, but separately for set and measured)
        self.assertEqual(data.p1_1_0.tolist(), [[1, 1], [2, 2]])
        self.assertEqual(data.p2_1.tolist(), [[3, 4]] * 2)

        self.assertEqual(data.p1_1_2_0.tolist(), [[[1, 1]] * 2, [[2, 2]] * 2])
        self.assertEqual(data.p2_2_1.tolist(), [[[3, 3], [4, 4]]] * 2)
        self.assertEqual(data.p3.tolist(), [[[5, 6]] * 2] * 2)

        # make sure rerunning this doesn't cause any problems
        data2 = loop.run_temp()
        keys2 = set(data.arrays.keys())
        self.assertEqual(keys, keys2)

    def test_repr(self):
        loop2 = Loop(self.p2[3:5:1], 0.001).each(self.p2)
        loop = Loop(self.p1[1:3:1], 0.001).each(self.p3,
                                                self.p2,
                                                loop2,
                                                self.p1)
        active_loop = loop
        data = active_loop.run_temp()
        expected = ('DataSet:\n'
                    '   mode     = DataMode.LOCAL\n'
                    '   location = False\n'
                    '   <Type>   | <array_id> | <array.name> | <array.shape>\n'
                    '   Setpoint | p1_set     | p1           | (2,)\n'
                    '   Measured | p3         | p3           | (2,)\n'
                    '   Measured | p2_1       | p2           | (2,)\n'
                    '   Setpoint | p2_set     | p2           | (2, 2)\n'
                    '   Measured | p2_2_0     | p2           | (2, 2)\n'
                    '   Measured | p1         | p1           | (2,)')
        self.assertEqual(data.__repr__(), expected)

    def test_default_measurement(self):
        self.p2.set(4)
        self.p3.set(5)

        data = Loop(self.p1[1:3:1], 0.001).run_temp()

        self.assertEqual(data.p1_set.tolist(), [1, 2])
        self.assertEqual(data.p2.tolist(), [4, 4])
        self.assertEqual(data.p3.tolist(), [5, 5])

        data = Loop(self.p1[1:3:1], 0.001).each(
            Loop(self.p2[3:5:1], 0.001)).run_temp()

        self.assertEqual(data.p1_set.tolist(), [1, 2])
        self.assertEqual(data.p2.tolist(), [[3, 4], [3, 4]])
        self.assertEqual(data.p2_set.tolist(), [[3, 4], [3, 4]])
        self.assertEqual(data.p3.tolist(), [[5, 5]] * 2)

    def test_tasks_callable_arguments(self):
        data = Loop(self.p1[1:3:1], 0.01).each(
            Task(self.p2.set, self.p1),
            Task(self.p3.set, self.p1.get),
            self.p2, self.p3).run_temp()

        self.assertEqual(data.p2.tolist(), [1, 2])
        self.assertEqual(data.p3.tolist(), [1, 2])

        def test_func(*args, **kwargs):
            self.assertEqual(args, (1, 2))
            self.assertEqual(kwargs, {'a_kwarg': 4})

        data = Loop(self.p1[1:2:1], 0.01).each(
            Task(self.p2.set, self.p1 * 2),
            Task(test_func, self.p1, self.p1 * 2, a_kwarg=self.p1 * 4),
            self.p2, self.p3).run_temp()

        self.assertEqual(data.p2.tolist(), [2])

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

        # give it a "process" as if it was run in the bg before,
        # check that this gets cleared
        loop.process = 'TDD'

        data = loop.run_temp()

        self.assertFalse(hasattr(loop, 'process'))

        self.assertEqual(data.p1_set.tolist(), [1, 2])
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
        self.assertEqual(data.p1_set.tolist(), [1, 2])
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
        # this one has names and shapes
        mg = MultiGetter(one=1, onetwo=(1, 2))
        self.assertTrue(hasattr(mg, 'names'))
        self.assertTrue(hasattr(mg, 'shapes'))
        self.assertEqual(mg.name, 'None')
        self.assertFalse(hasattr(mg, 'shape'))
        loop = Loop(self.p1[1:3:1], 0.001).each(mg)
        data = loop.run_temp()

        self.assertEqual(data.p1_set.tolist(), [1, 2])
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

        # this one has name and shape
        mg = MultiGetter(arr=(4, 5, 6))
        self.assertTrue(hasattr(mg, 'name'))
        self.assertTrue(hasattr(mg, 'shape'))
        self.assertFalse(hasattr(mg, 'names'))
        self.assertFalse(hasattr(mg, 'shapes'))
        loop = Loop(self.p1[1:3:1], 0.001).each(mg)
        data = loop.run_temp()

        self.assertEqual(data.p1_set.tolist(), [1, 2])
        self.assertEqual(data.arr.tolist(), [[4, 5, 6]] * 2)
        self.assertEqual(data.index0.tolist(), [[0, 1, 2]] * 2)

        mg = MultiGetter(arr2d=((21, 22), (23, 24)))
        loop = Loop(self.p1[1:3:1], 0.001).each(mg)
        data = loop.run_temp()

        self.assertEqual(data.p1_set.tolist(), [1, 2])
        self.assertEqual(data.arr2d.tolist(), [[[21, 22], [23, 24]]] * 2)
        self.assertEqual(data.index0.tolist(), [[0, 1]] * 2)
        self.assertEqual(data.index1.tolist(), [[[0, 1]] * 2] * 2)

    def test_bad_actors(self):
        def f():
            return 42

        class NoName:
            def get(self):
                return 42

        class HasName:
            def get(self):
                return 42

            name = 'IHazName!'

        class HasNames:
            def get(self):
                return 42

            names = 'Namezz'

        # first two minimal working gettables
        Loop(self.p1[1:3:1]).each(HasName())
        Loop(self.p1[1:3:1]).each(HasNames())

        for bad_action in (f, 42, NoName()):
            with self.assertRaises(TypeError):
                # include a good action too, just to make sure we look
                # at the whole list
                Loop(self.p1[1:3:1]).each(self.p1, bad_action)

        with self.assertRaises(ValueError):
            # invalid sweep values
            Loop(self.p1[-20:20:1]).each(self.p1)

    def test_very_short_delay(self):
        with LogCapture() as logs:
            Loop(self.p1[1:3:1], 1e-9).each(self.p1).run_temp()

        self.assertEqual(logs.value.count('negative delay'), 2, logs.value)

    def test_zero_delay(self):
        with LogCapture() as logs:
            Loop(self.p1[1:3:1]).each(self.p1).run_temp()

        self.assertEqual(logs.value.count('negative delay'), 0, logs.value)

    def test_breakif(self):
        nan = float('nan')
        loop = Loop(self.p1[1:6:1])
        data = loop.each(self.p1, BreakIf(self.p1 >= 3)).run_temp()
        self.assertEqual(repr(data.p1.tolist()),
                         repr([1., 2., 3., nan, nan]))

        data = loop.each(BreakIf(self.p1.get_latest >= 3), self.p1).run_temp()
        self.assertEqual(repr(data.p1.tolist()),
                         repr([1., 2., nan, nan, nan]))

        with self.assertRaises(TypeError):
            BreakIf(True)
        with self.assertRaises(TypeError):
            BreakIf(self.p1.set)

    def test_then_construction(self):
        loop = Loop(self.p1[1:6:1])
        task1 = Task(self.p1.set, 2)
        task2 = Wait(0.02)
        loop2 = loop.then(task1)
        loop3 = loop2.then(task2, task1)
        loop4 = loop3.then(task2, overwrite=True)
        loop5 = loop4.each(self.p1, BreakIf(self.p1 >= 3))
        loop6 = loop5.then(task1)
        loop7 = loop6.then(task1, overwrite=True)

        # original loop is untouched, same as .each and .loop
        self.assertEqual(loop.then_actions, ())

        # but loop2 has the task we asked for
        self.assertEqual(loop2.then_actions, (task1,))

        # loop3 gets the other tasks appended
        self.assertEqual(loop3.then_actions, (task1, task2, task1))

        # loop4 gets only the new one
        self.assertEqual(loop4.then_actions, (task2,))

        # tasks survive .each
        self.assertEqual(loop5.then_actions, (task2,))

        # and ActiveLoop.then works the same way as Loop.then
        self.assertEqual(loop6.then_actions, (task2, task1))
        self.assertEqual(loop7.then_actions, (task1,))

        # .then rejects Loops and others that are valid loop actions
        for action in (loop2, loop7, BreakIf(self.p1 >= 3), self.p1,
                       True, 42):
            with self.assertRaises(TypeError):
                loop.then(action)

    def check_snap_ts(self, container, key, ts_set):
        self.assertIn(container[key], ts_set)
        del container[key]

    def test_then_action(self):
        self.maxDiff = None
        nan = float('nan')
        self.p1.set(5)
        f_calls, g_calls = [], []

        def f():
            f_calls.append(1)

        def g():
            g_calls.append(1)

        breaker = BreakIf(self.p1 >= 3)
        ts1 = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # evaluate param snapshots now since later value will change
        p1snap = self.p1.snapshot()
        self.p2.set(2)
        p2snap = self.p2.snapshot()
        self.p3.set(3)
        p3snap = self.p3.snapshot()
        data = Loop(self.p1[1:6:1]).each(
            self.p1, breaker
        ).then(
            Task(self.p1.set, 2), Wait(0.01), Task(f)
        ).run_temp()
        ts2 = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        self.assertEqual(repr(data.p1.tolist()),
                         repr([1., 2., 3., nan, nan]))
        self.assertEqual(self.p1.get(), 2)
        self.assertEqual(len(f_calls), 1)

        # this loop makes use of all the features, so use it to test
        # DataSet metadata
        loopmeta = data.metadata['loop']
        default_meas_meta = data.metadata['station']['default_measurement']
        # assuming the whole loop takes < 1 sec, all timestamps
        # should each be the same as one of the bounding times
        self.check_snap_ts(loopmeta, 'ts_start', (ts1, ts2))
        self.check_snap_ts(loopmeta, 'ts_end', (ts1, ts2))
        self.check_snap_ts(loopmeta['sweep_values']['parameter'],
                           'ts', (ts1, ts2))
        self.check_snap_ts(loopmeta['actions'][0], 'ts', (ts1, ts2))
        self.check_snap_ts(default_meas_meta[0], 'ts', (ts1, ts2, None))
        self.check_snap_ts(default_meas_meta[1], 'ts', (ts1, ts2, None))
        del p1snap['ts'], p2snap['ts'], p3snap['ts']

        self.assertEqual(data.metadata, {
            'station': {
                'instruments': {},
                'parameters': {},
                'components': {},
                'default_measurement': [p2snap, p3snap]
            },
            'loop': {
                'background': False,
                'use_threads': False,
                'use_data_manager': False,
                '__class__': 'qcodes.loops.ActiveLoop',
                'sweep_values': {
                    'parameter': p1snap,
                    'values': [{'first': 1, 'last': 5, 'num': 5,
                               'type': 'linear'}]
                },
                'delay': 0,
                'actions': [p1snap, breaker.snapshot()],
                'then_actions': [
                    {'type': 'Task', 'func': repr(self.p1.set)},
                    {'type': 'Wait', 'delay': 0.01},
                    {'type': 'Task', 'func': repr(f)}
                ]
            }
        })

        # now test a nested loop with .then inside and outside
        f_calls[:] = []

        Loop(self.p1[1:3:1]).each(
            Loop(self.p2[1:3:1]).each(self.p2).then(Task(g))
        ).then(Task(f)).run_temp()

        self.assertEqual(len(f_calls), 1)
        self.assertEqual(len(g_calls), 2)

        # Loop.loop nesting always just makes the .then actions run after
        # the outer loop
        f_calls[:] = []
        Loop(self.p1[1:3:1]).then(Task(f)).loop(self.p2[1:3:1]).each(
            self.p1
        ).run_temp()
        self.assertEqual(len(f_calls), 1)

        f_calls[:] = []
        Loop(self.p1[1:3:1]).loop(self.p2[1:3:1]).then(Task(f)).each(
            self.p1
        ).run_temp()
        self.assertEqual(len(f_calls), 1)

        f_calls[:] = []
        Loop(self.p1[1:3:1]).loop(self.p2[1:3:1]).each(
            self.p1
        ).then(Task(f)).run_temp()
        self.assertEqual(len(f_calls), 1)


class AbortingGetter(ManualParameter):
    '''
    A manual parameter that can only be measured a couple of times
    before it aborts the loop that's measuring it.

    You have to attach the queue after construction with set_queue
    so you can grab it from the loop that uses the parameter.
    '''
    def __init__(self, *args, count=1, msg=None, **kwargs):
        self._count = self._initial_count = count
        self.msg = msg
        # also need a _signal_queue, but that has to be added later
        super().__init__(*args, **kwargs)

    def get(self):
        self._count -= 1
        if self._count <= 0:
            self._signal_queue.put(self.msg)
        return super().get()

    def set_queue(self, queue):
        self._signal_queue = queue

    def reset(self):
        self._count = self._initial_count


class TestSignal(TestCase):
    def test_halt(self):
        p1 = AbortingGetter('p1', count=2, vals=Numbers(-10, 10),
                            msg=ActiveLoop.HALT_DEBUG)
        loop = Loop(p1[1:6:1], 0.005).each(p1)
        # we want to test what's in data, so get it ahead of time
        # because loop.run will not return.
        data = loop.get_data_set(location=False)
        p1.set_queue(loop.signal_queue)

        with self.assertRaises(_DebugInterrupt):
            # need to use explicit loop.run rather than run_temp
            # so we can avoid providing location=False twice, which
            # is an error.
            loop.run(background=False, data_manager=False, quiet=True)

        self.check_data(data)

    def test_halt_quiet(self):
        p1 = AbortingGetter('p1', count=2, vals=Numbers(-10, 10),
                            msg=ActiveLoop.HALT)
        loop = Loop(p1[1:6:1], 0.005).each(p1)
        p1.set_queue(loop.signal_queue)

        # does not raise, just quits, but the data set looks the same
        # as in test_halt
        data = loop.run_temp()
        self.check_data(data)

    def check_data(self, data):
        nan = float('nan')
        self.assertEqual(data.p1.tolist()[:2], [1, 2])
        # when NaN is involved, I'll just compare reprs, because NaN!=NaN
        self.assertEqual(repr(data.p1.tolist()[-2:]), repr([nan, nan]))
        # because of the way the waits work out, we can get an extra
        # point measured before the interrupt is registered. But the
        # test would be valid either way.
        self.assertIn(repr(data.p1[2]), (repr(nan), repr(3), repr(3.0)))


class TestMetaData(TestCase):
    def test_basic(self):
        p1 = AbortingGetter('p1', count=2, vals=Numbers(-10, 10))
        sv = p1[1:3:1]
        loop = Loop(sv)

        # not sure why you'd do it, but you *can* snapshot a Loop
        expected = {
            '__class__': 'qcodes.loops.Loop',
            'sweep_values': sv.snapshot(),
            'delay': 0,
            'then_actions': []
        }
        self.assertEqual(loop.snapshot(), expected)
        loop = loop.then(Task(p1.set, 0), Wait(0.123))
        expected['then_actions'] = [
            {'type': 'Task', 'func': repr(p1.set)},
            {'type': 'Wait', 'delay': 0.123}
        ]

        # then test snapshot on an ActiveLoop
        breaker = BreakIf(p1.get_latest > 3)
        self.assertEqual(breaker.snapshot()['type'], 'BreakIf')
        # TODO: once we have reprs for DeferredOperations, test that
        # the right thing shows up in breaker.snapshot()['condition']
        loop = loop.each(p1, breaker)
        expected['__class__'] = 'qcodes.loops.ActiveLoop'
        expected['actions'] = [p1.snapshot(), breaker.snapshot()]

        self.assertEqual(loop.snapshot(), expected)
