from unittest import TestCase
from datetime import datetime, timedelta
import time

from qcodes.instrument.base import Instrument
from qcodes.instrument.mock import MockInstrument
from qcodes.instrument.parameter import Parameter, ManualParameter
from qcodes.instrument.function import Function
from qcodes.instrument.server import get_instrument_server_manager

from qcodes.utils.validators import Numbers, Ints, Strings, MultiType, Enum
from qcodes.utils.command import NoCommandError
from qcodes.utils.helpers import LogCapture
from qcodes.process.helpers import kill_processes

from .instrument_mocks import (AMockModel, MockInstTester,
                               MockGates, MockSource, MockMeter)
from .common import strip_qc


class TestParamConstructor(TestCase):
    def test_name_s(self):
        p = Parameter('simple')
        self.assertEqual(p.name, 'simple')

        with self.assertRaises(ValueError):
            # you need a name of some sort
            Parameter()

        # or names
        names = ['H1', 'L1']
        p = Parameter(names=names)
        self.assertEqual(p.names, names)
        self.assertFalse(hasattr(p, 'name'))

        # or both, that's OK too.
        names = ['Peter', 'Paul', 'Mary']
        p = Parameter(name='complex', names=names)
        self.assertEqual(p.names, names)
        # TODO: below seems wrong actually - we should let a parameter have
        # a simple name even if it has a names array. But then we need to
        # check everywhere this is used, and make sure everyone who cares
        # about it looks for names first.
        self.assertFalse(hasattr(p, 'name'))

        shape = (10,)
        setpoints = (range(10),)
        setpoint_names = ('my_sp',)
        setpoint_labels = ('A label!',)
        p = Parameter('makes_array', shape=shape, setpoints=setpoints,
                      setpoint_names=setpoint_names,
                      setpoint_labels=setpoint_labels)
        self.assertEqual(p.shape, shape)
        self.assertFalse(hasattr(p, 'shapes'))
        self.assertEqual(p.setpoints, setpoints)
        self.assertEqual(p.setpoint_names, setpoint_names)
        self.assertEqual(p.setpoint_labels, setpoint_labels)

        shapes = ((2,), (3,))
        setpoints = ((range(2),), (range(3),))
        setpoint_names = (('sp1',), ('sp2',))
        setpoint_labels = (('first label',), ('second label',))
        p = Parameter('makes arrays', shapes=shapes, setpoints=setpoints,
                      setpoint_names=setpoint_names,
                      setpoint_labels=setpoint_labels)
        self.assertEqual(p.shapes, shapes)
        self.assertFalse(hasattr(p, 'shape'))
        self.assertEqual(p.setpoints, setpoints)
        self.assertEqual(p.setpoint_names, setpoint_names)
        self.assertEqual(p.setpoint_labels, setpoint_labels)

    def test_repr(self):
        for i in [0, "foo", "", "fåil"]:
            with self.subTest(i=i):
                param = Parameter(name=i)
                s = param.__repr__()
                st = '<{}.{}: {} at {}>'.format(
                    param.__module__, param.__class__.__name__,
                    param.name, id(param))
                self.assertEqual(s, st)


class GatesBadDelayType(MockGates):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_parameter('chan0bad', get_cmd='c0?',
                           set_cmd=self.slow_neg_set,
                           get_parser=float,
                           vals=Numbers(-10, 10), step=0.2,
                           delay=0.01,
                           max_delay='forever')


class GatesBadDelayValue(MockGates):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_parameter('chan0bad', get_cmd='c0?',
                           set_cmd=self.slow_neg_set,
                           get_parser=float,
                           vals=Numbers(-10, 10), step=0.2,
                           delay=0.05,
                           max_delay=0.03)


class TestParameters(TestCase):
    def setUp(self):
        self.model = AMockModel()

        self.gates = MockGates(model=self.model)
        self.source = MockSource(model=self.model)
        self.meter = MockMeter(model=self.model, keep_history=False)

        self.init_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def tearDown(self):
        try:
            self.model.close()
            for instrument in [self.gates, self.source, self.meter]:
                instrument.close()
        except:
            pass

    def test_unpicklable(self):
        self.assertEqual(self.gates.add5(6), 11)
        # compare docstrings to make sure we're really calling add5
        # on the server, and seeing its docstring
        self.assertIn('The class copy of this should not get run',
                      MockInstTester.add5.__doc__)
        self.assertIn('not the same function as the original method',
                      self.gates.add5.__doc__)

    def test_slow_set(self):
        # at least for now, need a local instrument to test logging
        gatesLocal = MockGates(model=self.model, server_name=None)
        for param, logcount in (('chan0slow', 2), ('chan0slow2', 2),
                                ('chan0slow3', 0), ('chan0slow4', 1),
                                ('chan0slow5', 0)):
            gatesLocal.chan0.set(-0.5)

            with LogCapture() as logs:
                if param in ('chan0slow', 'chan0slow2', 'chan0slow3'):
                    # these are the stepped parameters
                    gatesLocal.set(param, 0.5)
                else:
                    # these are the non-stepped parameters that
                    # still have delays
                    gatesLocal.set(param, -1)
                    gatesLocal.set(param, 1)

            loglines = logs.value.split('\n')[:-1]
            # TODO: occasional extra negative delays here
            self.assertEqual(len(loglines), logcount, (param, logs.value))
            for line in loglines:
                self.assertTrue(line.startswith('negative delay'), line)

    def test_max_delay_errors(self):
        with self.assertRaises(TypeError):
            # add_parameter works remotely with string commands, but
            # function commands are not going to be picklable, since they
            # need to talk to the hardware, so these need to be included
            # from the beginning when the instrument is created on the
            # server.
            GatesBadDelayType(model=self.model)

        with self.assertRaises(ValueError):
            GatesBadDelayValue(model=self.model)

    def check_ts(self, ts_str):
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.assertTrue(self.init_ts <= ts_str <= now)

    def test_instances(self):
        instruments = [self.gates, self.source, self.meter]
        for instrument in instruments:
            for other_instrument in instruments:
                instances = instrument.instances()
                if other_instrument is instrument:
                    self.assertIn(instrument, instances)
                else:
                    self.assertNotIn(other_instrument, instances)

        # somehow instances never go away... there are always 3
        # extra references to every instrument object, so del doesn't
        # work. For this reason, instrument tests should take
        # the *last* instance to test.
        # so we can't test that the list of defined instruments is actually
        # *only* what we want to see defined.

    def test_mock_instrument(self):
        gates, source, meter = self.gates, self.source, self.meter

        # initial state
        # short form of getter
        self.assertEqual(gates.get('chan0'), 0)
        # shortcut to the parameter, longer form of get
        self.assertEqual(gates['chan0'].get(), 0)
        # explicit long form of getter
        self.assertEqual(gates.parameters['chan0'].get(), 0)
        # all 3 should produce the same history entry
        hist = gates.getattr('history')
        self.assertEqual(len(hist), 3)
        for item in hist:
            self.assertEqual(item[1:], ('ask', 'c0'))

        # errors trying to set (or validate) invalid param values
        # put here so we ensure that these errors don't make it to
        # the history (ie they don't result in hardware commands)
        with self.assertRaises(TypeError):
            gates.set('chan1', '1')
        with self.assertRaises(TypeError):
            gates.parameters['chan1'].validate('1')

        # change one param at a time
        gates.set('chan0', 0.5)
        self.assertEqual(gates.get('chan0'), 0.5)
        self.assertEqual(meter.get('amplitude'), 0.05)

        gates.set('chan1', 2)
        self.assertEqual(gates.get('chan1'), 2)
        self.assertEqual(meter.get('amplitude'), 0.45)

        gates.set('chan2', -3.2)
        self.assertEqual(gates.get('chan2'), -3.2)
        self.assertEqual(meter.get('amplitude'), -2.827)

        source.set('amplitude', 0.6)
        self.assertEqual(source.get('amplitude'), 0.6)
        self.assertEqual(meter.get('amplitude'), -16.961)

        gatehist = gates.getattr('history')
        sourcehist = source.getattr('history')
        meterhist = meter.getattr('history')
        # check just the size and timestamps of histories
        for entry in gatehist + sourcehist + meterhist:
            self.check_ts(entry[0])
        self.assertEqual(len(gatehist), 9)
        self.assertEqual(len(sourcehist), 5)
        # meter does not keep history but should still have a history attr
        self.assertEqual(len(meterhist), 0)

        # plus enough setters to check the parameter sweep
        # first source has to get the starting value
        self.assertEqual(sourcehist[0][1:], ('ask', 'ampl'))
        # then it writes each
        self.assertEqual(sourcehist[1][1:], ('write', 'ampl', '0.3000'))
        self.assertEqual(sourcehist[2][1:], ('write', 'ampl', '0.5000'))
        self.assertEqual(sourcehist[3][1:], ('write', 'ampl', '0.6000'))

        source.set('amplitude', 0.8)
        self.assertEqual(source.get('amplitude'), 0.8)
        gates.set('chan1', -2)
        self.assertEqual(gates.get('chan1'), -2)

        # test functions
        self.assertEqual(meter.call('echo', 1.2345), 1.23)  # model returns .2f
        # too many ways to do this...
        self.assertEqual(meter.echo.call(1.2345), 1.23)
        self.assertEqual(meter.echo(1.2345), 1.23)
        self.assertEqual(meter['echo'].call(1.2345), 1.23)
        self.assertEqual(meter['echo'](1.2345), 1.23)
        with self.assertRaises(TypeError):
            meter.call('echo', 1, 2)
        with self.assertRaises(TypeError):
            meter.call('echo', '1')

        # validating before actually trying to call
        with self.assertRaises(TypeError):
            meter.functions['echo'].validate(1, 2)
        with self.assertRaises(TypeError):
            meter.functions['echo'].validate('1')
        gates.call('reset')
        self.assertEqual(gates.get('chan0'), 0)

        self.assertEqual(meter.call('echo', 4.567), 4.57)
        gates.set('chan0', 1)
        self.assertEqual(gates.get('chan0'), 1)
        gates.call('reset')
        self.assertEqual(gates.get('chan0'), 0)

    def test_mock_set_sweep(self):
        gates = self.gates
        gates.set('chan0step', 0.5)
        gatehist = gates.getattr('history')
        self.assertEqual(len(gatehist), 6)
        self.assertEqual(
            [float(h[3]) for h in gatehist if h[1] == 'write'],
            [0.1, 0.2, 0.3, 0.4, 0.5])

    def test_mock_instrument_errors(self):
        gates, meter = self.gates, self.meter
        with self.assertRaises(ValueError):
            gates.ask('no question')
        with self.assertRaises(ValueError):
            gates.ask('question?yes but more after')

        with self.assertRaises(ValueError):
            gates.ask('ampl?')

        with self.assertRaises(TypeError):
            MockInstrument('', delay='forever')
        with self.assertRaises(TypeError):
            MockInstrument('', delay=-1)

        # TODO: when an error occurs during constructing an instrument,
        # we don't have the instrument but its server doesn't know to stop.
        # should figure out a way to remove it. (I thought I had but it
        # doesn't seem to have worked...)
        get_instrument_server_manager('MockInstruments').close()
        time.sleep(0.5)

        with self.assertRaises(AttributeError):
            MockInstrument('', model=None)

        with self.assertRaises(KeyError):
            gates.add_parameter('chan0', get_cmd='boo')
        with self.assertRaises(KeyError):
            gates.add_function('reset', call_cmd='hoo')

        with self.assertRaises(NotImplementedError):
            meter.set('amplitude', 0.5)
        meter.add_parameter('gain', set_cmd='gain {:.3f}')
        with self.assertRaises(NotImplementedError):
            meter.get('gain')

        with self.assertRaises(TypeError):
            gates.add_parameter('fugacity', set_cmd='f {:.4f}', vals=[1, 2, 3])

        # TODO: when an error occurs during constructing an instrument,
        # we don't have the instrument but its server doesn't know to stop.
        # should figure out a way to remove it. (I thought I had but it
        # doesn't seem to have worked...)
        kill_processes()

    def check_set_amplitude2(self, val, log_count, history_count):
        source = self.sourceLocal
        with LogCapture() as logs:
            source.amplitude2.set(val)

        loglines = logs.value.split('\n')[:-1]

        self.assertEqual(len(loglines), log_count, logs.value)
        for line in loglines:
            self.assertIn('cannot sweep', line.lower())
        hist = source.getattr('history')
        self.assertEqual(len(hist), history_count)

    def test_sweep_steps_edge_case(self):
        # MultiType with sweeping is weird - not sure why one would do this,
        # but we should handle it
        # at least for now, need a local instrument to check logging
        source = self.sourceLocal = MockSource(model=self.model,
                                               server_name=None)
        source.add_parameter('amplitude2', get_cmd='ampl?',
                             set_cmd='ampl:{}', get_parser=float,
                             vals=MultiType(Numbers(0, 1), Strings()),
                             step=0.2, delay=0.02)
        self.assertEqual(len(source.getattr('history')), 0)

        # 2 history items - get then set, and one warning (cannot sweep
        # number to string value)
        self.check_set_amplitude2('Off', log_count=1, history_count=2)

        # one more history item - single set, and one warning (cannot sweep
        # string to number)
        self.check_set_amplitude2(0.2, log_count=1, history_count=3)

        # the only real sweep (0.2 to 0.8) adds 3 set's to history and no logs
        self.check_set_amplitude2(0.8, log_count=0, history_count=6)

        # single set added to history, and another sweep warning num->string
        self.check_set_amplitude2('Off', log_count=1, history_count=7)

    def test_set_sweep_errors(self):
        gates = self.gates

        # for reference, some add_parameter's that should work
        gates.add_parameter('t0', set_cmd='{}', vals=Numbers(),
                            step=0.1, delay=0.01)
        gates.add_parameter('t2', set_cmd='{}', vals=Ints(),
                            step=1, delay=0.01,
                            max_val_age=0)

        with self.assertRaises(TypeError):
            # can't sweep non-numerics
            gates.add_parameter('t1', set_cmd='{}', vals=Strings(),
                                step=1, delay=0.01)
        with self.assertRaises(TypeError):
            # need a numeric step too
            gates.add_parameter('t1', set_cmd='{}', vals=Numbers(),
                                step='a skosh', delay=0.01)
        with self.assertRaises(TypeError):
            # Ints requires and int step
            gates.add_parameter('t1', set_cmd='{}', vals=Ints(),
                                step=0.1, delay=0.01)
        with self.assertRaises(ValueError):
            # need a non-negative step
            gates.add_parameter('t1', set_cmd='{}', vals=Numbers(),
                                step=-0.1, delay=0.01)
        with self.assertRaises(TypeError):
            # need a numeric delay
            gates.add_parameter('t1', set_cmd='{}', vals=Numbers(),
                                step=0.1, delay='a tad')
        with self.assertRaises(ValueError):
            # need a non-negative delay
            gates.add_parameter('t1', set_cmd='{}', vals=Numbers(),
                                step=0.1, delay=-0.01)
        with self.assertRaises(TypeError):
            # need a numeric max_val_age
            gates.add_parameter('t1', set_cmd='{}', vals=Numbers(),
                                step=0.1, delay=0.01,
                                max_val_age='an hour')
        with self.assertRaises(ValueError):
            # need a non-negative max_val_age
            gates.add_parameter('t1', set_cmd='{}', vals=Numbers(),
                                step=0.1, delay=0.01,
                                max_val_age=-1)

    def getmem(self, key):
        return self.source.ask('mem{}?'.format(key))

    def test_val_mapping(self):
        gates = self.gates

        # memraw has no mappings - it just sets and gets what the instrument
        # uses to encode this parameter
        gates.add_parameter('memraw', set_cmd='mem0:{}', get_cmd='mem0?',
                            vals=Enum('zero', 'one'))

        # memcoded maps the instrument codes ('zero' and 'one') into nicer
        # user values 0 and 1
        gates.add_parameter('memcoded', set_cmd='mem0:{}', get_cmd='mem0?',
                            val_mapping={0: 'zero', 1: 'one'})

        gates.memcoded.set(0)
        self.assertEqual(gates.memraw.get(), 'zero')
        self.assertEqual(gates.memcoded.get(), 0)
        self.assertEqual(self.getmem(0), 'zero')

        gates.memraw.set('one')
        self.assertEqual(gates.memcoded.get(), 1)
        self.assertEqual(gates.memraw.get(), 'one')
        self.assertEqual(self.getmem(0), 'one')

        with self.assertRaises(ValueError):
            gates.memraw.set(0)

        with self.assertRaises(ValueError):
            gates.memcoded.set('zero')

    def test_bare_function(self):
        # not a use case we want to promote, but it's there...
        p = ManualParameter('test')

        def doubler(x):
            p.set(x * 2)

        f = Function('f', call_cmd=doubler, args=[Numbers(-10, 10)])

        f(4)
        self.assertEqual(p.get(), 8)
        with self.assertRaises(ValueError):
            f(20)

    def test_standard_snapshot(self):
        self.maxDiff = None
        snap = self.meter.snapshot()
        strip_qc(snap)
        for psnap in snap['parameters'].values():
            strip_qc(psnap)

        self.assertEqual(snap, {
            '__class__': 'tests.instrument_mocks.MockMeter',
            'name': 'meter',
            'parameters': {
                'IDN': {
                    '__class__': (
                        'qcodes.instrument.parameter.StandardParameter'),
                    'instrument': 'tests.instrument_mocks.MockMeter',
                    'instrument_name': 'meter',
                    'label': 'IDN',
                    'name': 'IDN',
                    'ts': None,
                    'units': '',
                    'value': None
                },
                'amplitude': {
                    '__class__': (
                        'qcodes.instrument.parameter.StandardParameter'),
                    'instrument': 'tests.instrument_mocks.MockMeter',
                    'instrument_name': 'meter',
                    'label': 'amplitude',
                    'name': 'amplitude',
                    'ts': None,
                    'units': '',
                    'value': None
                }
            },
            'functions': {'echo': {}}
        })

        ampsnap = self.meter.snapshot(update=True)['parameters']['amplitude']
        amp = self.meter.get('amplitude')
        self.assertEqual(ampsnap['value'], amp)
        amp_ts = datetime.strptime(ampsnap['ts'], '%Y-%m-%d %H:%M:%S')
        self.assertLessEqual(amp_ts, datetime.now())
        self.assertGreater(amp_ts, datetime.now() - timedelta(seconds=1.1))

    def test_manual_snapshot(self):
        self.source.add_parameter('noise', parameter_class=ManualParameter)
        noise = self.source.noise

        noisesnap = self.source.snapshot()['parameters']['noise']
        strip_qc(noisesnap)
        self.assertEqual(noisesnap, {
            '__class__': 'qcodes.instrument.parameter.ManualParameter',
            'instrument': 'tests.instrument_mocks.MockSource',
            'instrument_name': 'source',
            'label': 'noise',
            'name': 'noise',
            'ts': None,
            'units': '',
            'value': None
        })

        noise.set(100)
        noisesnap = self.source.snapshot()['parameters']['noise']
        self.assertEqual(noisesnap['value'], 100)

        noise_ts = datetime.strptime(noisesnap['ts'], '%Y-%m-%d %H:%M:%S')
        self.assertLessEqual(noise_ts, datetime.now())
        self.assertGreater(noise_ts, datetime.now() - timedelta(seconds=1.1))

    def tests_get_latest(self):
        self.source.add_parameter('noise', parameter_class=ManualParameter)
        noise = self.source.noise

        self.assertIsNone(noise.get_latest())

        noise.set(100)

        mock_ts = datetime(2000, 3, 4)
        ts_str = mock_ts.strftime('%Y-%m-%d %H:%M:%S')
        noise.setattr('_latest_ts', mock_ts)
        self.assertEqual(noise.snapshot()['ts'], ts_str)

        self.assertEqual(noise.get_latest(), 100)
        self.assertEqual(noise.get_latest.get(), 100)

        # get_latest should not update ts
        self.assertEqual(noise.snapshot()['ts'], ts_str)

        # get_latest is not settable
        with self.assertRaises(AttributeError):
            noise.get_latest.set(50)

    def test_base_instrument_errors(self):
        b = Instrument('silent', server_name=None)

        with self.assertRaises(NotImplementedError):
            b.write('hello!')
        with self.assertRaises(NotImplementedError):
            b.ask('how are you?')

        with self.assertRaises(TypeError):
            b.add_function('skip', call_cmd='skip {}',
                           args=['not a validator'])
        with self.assertRaises(NoCommandError):
            b.add_function('jump')
        with self.assertRaises(NoCommandError):
            b.add_parameter('height')

    def test_manual_parameter(self):
        self.source.add_parameter('bias_resistor',
                                  parameter_class=ManualParameter,
                                  initial_value=1000)
        res = self.source.bias_resistor
        self.assertEqual(res.get(), 1000)

        res.set(1e9)
        self.assertEqual(res.get(), 1e9)
        # default vals is all numbers
        res.set(-1)
        self.assertEqual(res.get(), -1)

        self.source.add_parameter('alignment',
                                  parameter_class=ManualParameter,
                                  vals=Enum('lawful', 'neutral', 'chaotic'))
        alignment = self.source.alignment

        # a ManualParameter can have initial_value=None (default) even if
        # that's not a valid value to set later
        self.assertIsNone(alignment.get())
        with self.assertRaises(ValueError):
            alignment.set(None)

        alignment.set('lawful')
        self.assertEqual(alignment.get(), 'lawful')

        # None is the only invalid initial_value you can use
        with self.assertRaises(TypeError):
            self.source.add_parameter('alignment2',
                                      parameter_class=ManualParameter,
                                      initial_value='nearsighted')

    def test_deferred_ops(self):
        gates = self.gates
        c0, c1, c2 = gates.chan0, gates.chan1, gates.chan2

        c0.set(0)
        c1.set(1)
        c2.set(2)

        self.assertEqual((c0 + c1 + c2)(), 3)
        self.assertEqual((10 + (c0**2) + (c1**2) + (c2**2))(), 15)

        d = c1.get_latest / c0.get_latest
        with self.assertRaises(ZeroDivisionError):
            d()


class TestAttrAccess(TestCase):
    def tearDown(self):
        self.instrument.close()
        # do it twice - should not error, though the second is irrelevant
        self.instrument.close()

    def test_server(self):
        instrument = Instrument(name='test_server', server_name='attr_test')
        self.instrument = instrument

        # set one attribute with nested levels
        instrument.setattr('d1', {'a': {1: 2}})

        # get the whole dict
        self.assertEqual(instrument.getattr('d1'), {'a': {1: 2}})
        self.assertEqual(instrument.getattr('d1', 55), {'a': {1: 2}})

        # get parts
        self.assertEqual(instrument.getattr('d1["a"]'), {1: 2})
        self.assertEqual(instrument.getattr("d1['a'][1]"), 2)
        self.assertEqual(instrument.getattr('d1["a"][1]', 3), 2)

        # add an attribute inside, then delete it again
        instrument.setattr('d1["a"][2]', 23)
        self.assertEqual(instrument.getattr('d1'), {'a': {1: 2, 2: 23}})
        instrument.delattr('d1["a"][2]')
        self.assertEqual(instrument.getattr('d1'), {'a': {1: 2}})

        # test restarting the InstrumentServer - this clears these attrs
        instrument._manager.restart()
        self.assertIsNone(instrument.getattr('d1', None))


class TestLocalMock(TestCase):
    def setUp(self):
        self.model = AMockModel()

        self.gates = MockGates(self.model, server_name=None)
        self.source = MockSource(self.model, server_name=None)
        self.meter = MockMeter(self.model, server_name=None)

    def tearDown(self):
        self.model.close()
        for instrument in [self.gates, self.source, self.meter]:
            instrument.close()

    def test_local(self):
        self.gates.chan1.set(3.33)
        self.assertEqual(self.gates.chan1.get(), 3.33)

        self.gates.reset()
        self.assertEqual(self.gates.chan1.get(), 0)

        with self.assertRaises(ValueError):
            self.gates.ask('knock knock? Oh never mind.')
