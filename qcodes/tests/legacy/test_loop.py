from datetime import datetime
import time
from unittest import TestCase
import numpy as np
from unittest.mock import patch
import os

from qcodes.loops import Loop
from qcodes.actions import Task, Wait, BreakIf, _QcodesBreak
from qcodes.station import Station
from qcodes.data.data_array import DataArray
from qcodes.instrument.parameter import Parameter, MultiParameter
from qcodes.utils.validators import Numbers
from qcodes.logger.logger import LogCapture

from ..instrument_mocks import MultiGetter, DummyInstrument


class NanReturningParameter(MultiParameter):

    def __init__(self, name, instrument, names=('first', 'second'),
                 shapes=((), ())):

        super().__init__(name=name, names=names, shapes=shapes,
                         instrument=instrument)

    def get_raw(self):  # this results in a nan-filled DataArray
        return (13,)


class TestLoop(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p1 = Parameter('p1', get_cmd=None, set_cmd=None, vals=Numbers(-10, 10))
        cls.p2 = Parameter('p2', get_cmd=None, set_cmd=None,  vals=Numbers(-10, 10))
        cls.p3 = Parameter('p3', get_cmd=None, set_cmd=None,  vals=Numbers(-10, 10))
        cls.instr = DummyInstrument('dummy_bunny')
        cls.p4_crazy = NanReturningParameter('p4_crazy', instrument=cls.instr)
        Station()

    @classmethod
    def tearDownClass(cls):
        cls.instr.close()

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
                    '   location = False\n'
                    '   <Type>   | <array_id> | <array.name> | <array.shape>\n'
                    '   Setpoint | p1_set     | p1           | (2,)\n'
                    '   Measured | p3         | p3           | (2,)\n'
                    '   Measured | p2_1       | p2           | (2,)\n'
                    '   Setpoint | p2_set     | p2           | (2, 2)\n'
                    '   Measured | p2_2_0     | p2           | (2, 2)\n'
                    '   Measured | p1         | p1           | (2,)')
        self.assertEqual(data.__repr__(), expected)

    def test_measurement_with_many_nans(self):
        loop = Loop(self.p1.sweep(0, 1, num=10),
                    delay=0.05).each(self.p4_crazy)
        ds = loop.get_data_set()
        loop.run()

        # assert that both the snapshot and the datafile are there
        self.assertEqual(len(os.listdir(ds.location)), 2)

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
            Task(self.p2.set, lambda: self.p1.get() * 2),
            Task(test_func, self.p1, lambda: self.p1.get() * 2,
                 a_kwarg=lambda: self.p1.get() * 4),
            self.p2, self.p3).run_temp()

        self.assertEqual(data.p2.tolist(), [2])

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
        # On slow CI machines, there can be a significant additional delay.
        # So what are we even testing here..?
        self.assertLessEqual(delay, 0.07)

    def test_composite_params(self):
        # this one has names and shapes
        mg = MultiGetter(one=1, onetwo=(1, 2))
        self.assertTrue(hasattr(mg, 'names'))
        self.assertTrue(hasattr(mg, 'shapes'))
        self.assertEqual(mg.name, 'multigetter')
        self.assertFalse(hasattr(mg, 'shape'))
        loop = Loop(self.p1[1:3:1], 0.001).each(mg)
        data = loop.run_temp()

        self.assertEqual(data.p1_set.tolist(), [1, 2])
        self.assertEqual(data.one.tolist(), [1, 1])
        self.assertEqual(data.onetwo.tolist(), [[1, 2]] * 2)
        self.assertEqual(data.index0_set.tolist(), [[0, 1]] * 2)

        # give it setpoints, names, and labels
        mg.setpoints = (None, ((10, 11),))
        sp_name = 'highest'
        mg.setpoint_names = (None, (sp_name,))
        sp_label = 'does it go to 11?'
        mg.setpoint_labels = (None, (sp_label,))

        data = loop.run_temp()

        self.assertEqual(data.highest_set.tolist(), [[10, 11]] * 2)
        self.assertEqual(data.highest_set.label, sp_label)

        # setpoints as DataArray - name and label here override
        # setpoint_names and setpoint_labels attributes
        new_sp_name = 'bgn'
        new_sp_label = 'boogie nights!'
        sp_dataarray = DataArray(preset_data=[6, 7], name=new_sp_name,
                                 label=new_sp_label)
        mg.setpoints = (None, (sp_dataarray,))

        data = loop.run_temp()
        self.assertEqual(data.bgn_set.tolist(), [[6, 7]] * 2)
        self.assertEqual(data.bgn_set.label, new_sp_label)

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

        # this one still has names and shapes
        mg = MultiGetter(arr=(4, 5, 6))
        self.assertTrue(hasattr(mg, 'name'))
        self.assertFalse(hasattr(mg, 'shape'))
        self.assertTrue(hasattr(mg, 'names'))
        self.assertTrue(hasattr(mg, 'shapes'))
        loop = Loop(self.p1[1:3:1], 0.001).each(mg)
        data = loop.run_temp()

        self.assertEqual(data.p1_set.tolist(), [1, 2])
        self.assertEqual(data.arr.tolist(), [[4, 5, 6]] * 2)
        self.assertEqual(data.index0_set.tolist(), [[0, 1, 2]] * 2)

        mg = MultiGetter(arr2d=((21, 22), (23, 24)))
        loop = Loop(self.p1[1:3:1], 0.001).each(mg)
        data = loop.run_temp()

        self.assertEqual(data.p1_set.tolist(), [1, 2])
        self.assertEqual(data.arr2d.tolist(), [[[21, 22], [23, 24]]] * 2)
        self.assertEqual(data.index0_set.tolist(), [[0, 1]] * 2)
        self.assertEqual(data.index1_set.tolist(), [[[0, 1]] * 2] * 2)

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
        data = loop.each(self.p1, BreakIf(lambda: self.p1.get() >= 3)).run_temp()
        self.assertEqual(repr(data.p1.tolist()),
                         repr([1., 2., 3., nan, nan]))

        data = loop.each(BreakIf(lambda: self.p1.get_latest.get() >= 3), self.p1).run_temp()
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
        loop5 = loop4.each(self.p1, BreakIf(lambda: self.p1.get() >= 3))
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
        for action in (loop2, loop7, BreakIf(lambda: self.p1() >= 3), self.p1,
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

        breaker = BreakIf(lambda: self.p1() >= 3)
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
        # assuming the whole loop takes < 1 sec, all timestamps
        # should each be the same as one of the bounding times
        self.check_snap_ts(loopmeta, 'ts_start', (ts1, ts2))
        self.check_snap_ts(loopmeta, 'ts_end', (ts1, ts2))
        self.check_snap_ts(loopmeta['sweep_values']['parameter'],
                           'ts', (ts1, ts2))
        self.check_snap_ts(loopmeta['actions'][0], 'ts', (ts1, ts2))
        del p1snap['ts'], p2snap['ts'], p3snap['ts']

        self.assertEqual(data.metadata, {
            'station': {
                'instruments': {},
                'parameters': {},
                'components': {},
                'config': None,
            },
            'loop': {
                'use_threads': False,
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


class AbortingGetter(Parameter):
    """
    A manual parameter that can only be measured n times
    before it aborts the loop that's measuring it.
    """
    def __init__(self, *args, count=1, msg=None, **kwargs):
        self._count = self._initial_count = count
        # also need a _signal_queue, but that has to be added later
        super().__init__(*args, **kwargs)

    def get_raw(self):
        self._count -= 1
        if self._count <= 0:
            raise _QcodesBreak
        return self.cache.raw_value

    def reset(self):
        self._count = self._initial_count


class Test_halt(TestCase):
    def test_halt(self):
        abort_after = 3
        self.res = list(np.arange(0, abort_after-1, 1.))
        [self.res.append(float('nan')) for i in range(0, abort_after-1)]

        p1 = AbortingGetter('p1', count=abort_after, vals=Numbers(-10, 10), set_cmd=None)
        loop = Loop(p1.sweep(0, abort_after, 1), 0.005).each(p1)
        # we want to test what's in data, so get it ahead of time
        # because loop.run will not return.
        data = loop.get_data_set(location=False)

        loop.run(quiet=True)
        self.assertEqual(repr(data.p1.tolist()), repr(self.res))


class TestMetaData(TestCase):
    def test_basic(self):
        p1 = AbortingGetter('p1', count=2, vals=Numbers(-10, 10), set_cmd=None)
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
        breaker = BreakIf(lambda: p1.get_latest() > 3)
        self.assertEqual(breaker.snapshot()['type'], 'BreakIf')
        loop = loop.each(p1, breaker)
        expected['__class__'] = 'qcodes.loops.ActiveLoop'
        expected['actions'] = [p1.snapshot(), breaker.snapshot()]

        self.assertEqual(loop.snapshot(), expected)
