"""
Test suite for  instument.*
"""
from datetime import datetime, timedelta
from unittest import TestCase
import time

from qcodes.instrument.base import Instrument
from qcodes.instrument.mock import MockInstrument
from qcodes.instrument.parameter import ManualParameter
from qcodes.instrument.server import get_instrument_server_manager

from qcodes.utils.validators import Numbers, Ints, Strings, MultiType, Enum
from qcodes.utils.command import NoCommandError
from qcodes.utils.helpers import LogCapture
from qcodes.process.helpers import kill_processes

from .instrument_mocks import (AMockModel, MockInstTester,
                               MockGates, MockSource, MockMeter,
                               DummyInstrument)
from .common import strip_qc


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


class TestInstrument(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = AMockModel()

        cls.gates = MockGates(model=cls.model, server_name='')
        cls.source = MockSource(model=cls.model, server_name='')
        cls.meter = MockMeter(model=cls.model, keep_history=False, server_name='')

    def setUp(self):
        # reset the model state via the gates function
        self.gates.reset()

        # then reset each instrument's state, so we can avoid the time to
        # completely reinstantiate with every test case
        for inst in (self.gates, self.source, self.meter):
            inst.restart()
        self.init_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    @classmethod
    def tearDownClass(cls):
        try:
            cls.model.close()
            for instrument in [cls.gates, cls.source, cls.meter]:
                instrument.close()
                # do it twice - should not error, though the second is
                # irrelevant
                instrument.close()
        except:
            pass

        # TODO: when an error occurs during constructing an instrument,
        # we don't have the instrument but its server doesn't know to stop.
        # should figure out a way to remove it. (I thought I had but it
        # doesn't seem to have worked...)
        # for test_mock_instrument_errors
        kill_processes()

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
        gatesLocal = MockGates(model=self.model, server_name=None,
                               name='gateslocal')
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
            GatesBadDelayType(model=self.model, name='gatesBDT')

        with self.assertRaises(ValueError):
            GatesBadDelayValue(model=self.model, name='gatesBDV')

    def check_ts(self, ts_str):
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.assertTrue(self.init_ts <= ts_str <= now)

    def test_instances(self):
        instruments = [self.gates, self.source, self.meter]
        for instrument in instruments:
            for other_instrument in instruments:
                instances = instrument.instances()
                # check that each instrument is in only its own
                # instances list
                # also test type checking in find_instrument,
                # but we need to use find_component so it executes
                # on the server
                if other_instrument is instrument:
                    self.assertIn(instrument, instances)

                    name2 = other_instrument.find_component(
                        instrument.name + '.name',
                        other_instrument._instrument_class)
                    self.assertEqual(name2, instrument.name)
                else:
                    self.assertNotIn(other_instrument, instances)

                    with self.assertRaises(TypeError):
                        other_instrument.find_component(
                            instrument.name + '.name',
                            other_instrument._instrument_class)

                # check that we can find each instrument from any other
                # find_instrument is explicitly mapped in RemoteInstrument
                # so this call gets executed in the main process
                self.assertEqual(
                    instrument,
                    other_instrument.find_instrument(instrument.name))

                # but find_component is not, so it executes on the server
                self.assertEqual(
                    instrument.name,
                    other_instrument.find_component(instrument.name + '.name'))

            # check that we can find this instrument from the base class
            self.assertEqual(instrument,
                             Instrument.find_instrument(instrument.name))

        # somehow instances never go away... there are always 3
        # extra references to every instrument object, so del doesn't
        # work. For this reason, instrument tests should take
        # the *last* instance to test.
        # so we can't test that the list of defined instruments is actually
        # *only* what we want to see defined.

    def test_instance_name_uniqueness(self):
        with self.assertRaises(KeyError):
            MockGates(model=self.model)

    def test_remove_instance(self):
        self.gates.close()
        self.assertEqual(self.gates.instances(), [])
        with self.assertRaises(KeyError):
            Instrument.find_instrument('gates')

        type(self).gates = MockGates(model=self.model, server_name="")
        self.assertEqual(self.gates.instances(), [self.gates])
        self.assertEqual(Instrument.find_instrument('gates'), self.gates)

    def test_creation_failure(self):
        # this we already know should fail (see test_max_delay_errors)
        name = 'gatesFailing'
        with self.assertRaises(ValueError):
            GatesBadDelayValue(model=self.model, name=name, server_name='')

        # this instrument should not be in the instance list
        with self.assertRaises(KeyError):
            Instrument.find_instrument(name)

        # now do the same with a local instrument
        name = 'gatesFailing2'
        with self.assertRaises(ValueError):
            GatesBadDelayValue(model=self.model, name=name, server_name=None)

        # this instrument should not be in the instance list
        with self.assertRaises(KeyError):
            Instrument.find_instrument(name)

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

    def test_mock_idn(self):
        self.assertEqual(self.gates.IDN(), {
            'vendor': None,
            'model': 'MockGates',
            'serial': 'gates',
            'firmware': None
        })

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
            MockInstrument('mockbaddelay1', delay='forever')
        with self.assertRaises(TypeError):
            # TODO: since this instrument didn't work, it should be OK
            # to use the same name again... how do we allow that?
            MockInstrument('mockbaddelay2', delay=-1)

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
                                               server_name=None,
                                               name='sourcelocal')
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

    def test_val_mapping_ints(self):
        gates = self.gates

        gates.add_parameter('moderaw', set_cmd='mem0:{}', get_cmd='mem0?',
                            vals=Enum('0', '1'))

        # modecoded maps the instrument codes ('0' and '1') into nicer
        # user values 'AC' and 'DC'
        # Here we're using integers in the mapping, rather than turning
        # them into strings.
        gates.add_parameter('modecoded', set_cmd='mem0:{}', get_cmd='mem0?',
                            val_mapping={'DC': 0, 'AC': 1})

        gates.modecoded.set('AC')
        self.assertEqual(gates.moderaw.get(), '1')
        self.assertEqual(gates.modecoded.get(), 'AC')
        self.assertEqual(self.getmem(0), '1')

        gates.moderaw.set('0')
        self.assertEqual(gates.modecoded.get(), 'DC')
        self.assertEqual(gates.moderaw.get(), '0')
        self.assertEqual(self.getmem(0), '0')

        with self.assertRaises(ValueError):
            gates.modecoded.set(0)

        with self.assertRaises(ValueError):
            gates.modecoded.set('0')

        with self.assertRaises(ValueError):
            gates.moderaw.set('DC')

    def test_val_mapping_parsers(self):
        gates = self.gates

        gates.add_parameter('moderaw', set_cmd='mem0:{}', get_cmd='mem0?',
                            vals=Enum('0', '1'))

        with self.assertRaises(TypeError):
            # set_parser is not allowed with val_mapping
            gates.add_parameter('modecoded', set_cmd='mem0:{}',
                                get_cmd='mem0?',
                                val_mapping={'DC': 0, 'AC': 1},
                                set_parser=float)

        gates.add_parameter('modecoded', set_cmd='mem0:{:.0f}',
                            get_cmd='mem0?',
                            val_mapping={'DC': 0.0, 'AC': 1.0},
                            get_parser=float)

        gates.modecoded.set('AC')
        self.assertEqual(gates.moderaw.get(), '1')
        self.assertEqual(gates.modecoded.get(), 'AC')
        self.assertEqual(self.getmem(0), '1')

        gates.moderaw.set('0')
        self.assertEqual(gates.modecoded.get(), 'DC')
        self.assertEqual(gates.moderaw.get(), '0')
        self.assertEqual(self.getmem(0), '0')

        with self.assertRaises(ValueError):
            gates.modecoded.set(0)

        with self.assertRaises(ValueError):
            gates.modecoded.set('0')

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
        # set / get with __call__ shortcut
        res(-1)
        self.assertEqual(res(), -1)

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

    def test_attr_access(self):
        instrument = self.gates

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

    def test_component_attr_access(self):
        instrument = self.gates
        method = instrument.add5
        parameter = instrument.chan1
        function = instrument.reset

        # RemoteMethod objects have no attributes besides __doc__, so test
        # that this gets appropriately decorated
        self.assertIn('RemoteMethod add5 in RemoteInstrument', method.__doc__)
        # and also contains the remote doc
        self.assertIn('not the same function as the original method',
                      method.__doc__)

        # units is a remote attribute of parameters
        # this one is initially blank
        self.assertEqual(parameter.units, '')
        parameter.units = 'Smoots'
        self.assertEqual(parameter.units, 'Smoots')
        self.assertNotIn('units', parameter.__dict__)
        self.assertEqual(instrument.getattr(parameter.name + '.units'),
                         'Smoots')
        # we can delete it remotely, and this is reflected in dir()
        self.assertIn('units', dir(parameter))
        del parameter.units
        self.assertNotIn('units', dir(parameter))
        with self.assertRaises(AttributeError):
            parameter.units

        # and set it again, it's still remote.
        parameter.units = 'Furlongs per fortnight'
        self.assertIn('units', dir(parameter))
        self.assertEqual(parameter.units, 'Furlongs per fortnight')
        self.assertNotIn('units', parameter.__dict__)
        self.assertEqual(instrument.getattr(parameter.name + '.units'),
                         'Furlongs per fortnight')
        # we get the correct result if someone else sets it on the server
        instrument._write_server('setattr', parameter.name + '.units', 'T')
        self.assertEqual(parameter.units, 'T')
        self.assertEqual(parameter.getattr('units'), 'T')

        # attributes not specified as remote are local
        with self.assertRaises(AttributeError):
            parameter.something
        parameter.something = 42
        self.assertEqual(parameter.something, 42)
        self.assertEqual(parameter.__dict__['something'], 42)
        with self.assertRaises(AttributeError):
            instrument.getattr(parameter.name + '.something')
        with self.assertRaises(AttributeError):
            # getattr method is only for remote attributes
            parameter.getattr('something')
        self.assertIn('something', dir(parameter))
        del parameter.something
        self.assertNotIn('something', dir(parameter))
        with self.assertRaises(AttributeError):
            parameter.something

        # call a remote method
        self.assertEqual(set(parameter.callattr('get_attrs')),
                         parameter._attrs)

        # functions have remote attributes too
        self.assertEqual(function._args, [])
        self.assertNotIn('_args', function.__dict__)
        function._args = 'args!'
        self.assertEqual(function._args, 'args!')

        # a component with no docstring still gets the decoration
        foo = instrument.foo
        self.assertEqual(foo.__doc__,
                         'RemoteParameter foo in RemoteInstrument gates')

    def test_update_components(self):
        gates = self.gates

        gates.delattr('chan0.label')
        gates.setattr('chan0.cheese', 'gorgonzola')
        # we've altered the server copy, but not the RemoteParameter
        self.assertIn('label', gates.chan0._attrs)
        self.assertNotIn('cheese', gates.chan0._attrs)
        # keep a reference to the original chan0 RemoteParameter to make sure
        # it is still the same object later
        chan0_original = gates.chan0

        gates.update()

        self.assertIs(gates.chan0, chan0_original)
        # now the RemoteParameter should have the updates
        self.assertNotIn('label', gates.chan0._attrs)
        self.assertIn('cheese', gates.chan0._attrs)

    def test_add_delete_components(self):
        gates = self.gates

        # rather than call gates.add_parameter, which has a special proxy
        # on the remote so it updates the components immediately, we'll call
        # the server version directly
        attr_list = gates.callattr('add_parameter', 'chan0X', get_cmd='c0?',
                                   set_cmd='c0:{:.4f}', get_parser=float)
        gates.delattr('parameters["chan0"]')

        # the RemoteInstrument does not have these changes yet
        self.assertIn('chan0', gates.parameters)
        self.assertNotIn('chan0X', gates.parameters)

        gates.update()

        # now the RemoteInstrument has the changes
        self.assertNotIn('chan0', gates.parameters)
        self.assertIn('chan0X', gates.parameters)
        self.assertEqual(gates.chan0X._attrs, set(attr_list))

    def test_reprs(self):
        gates = self.gates
        self.assertIn(gates.name, repr(gates))
        self.assertIn('chan1', repr(gates.chan1))
        self.assertIn('reset', repr(gates.reset))

    def test_remote_sweep_values(self):
        chan1 = self.gates.chan1

        sv1 = chan1[1:4:1]
        self.assertEqual(len(sv1), 3)
        self.assertIn(2, sv1)

        sv2 = chan1.sweep(start=2, stop=3, num=6)
        self.assertEqual(len(sv2), 6)
        self.assertIn(2.2, sv2)

    def test_add_function(self):
        gates = self.gates
        # add a function remotely
        gates.add_function('reset2', call_cmd='rst')
        gates.chan1(4)
        self.assertEqual(gates.chan1(), 4)
        gates.reset2()
        self.assertEqual(gates.chan1(), 0)


class TestLocalMock(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = AMockModel()

        cls.gates = MockGates(model=cls.model, server_name=None)
        cls.source = MockSource(model=cls.model, server_name=None)
        cls.meter = MockMeter(model=cls.model, server_name=None)

    @classmethod
    def tearDownClass(cls):
        cls.model.close()
        for instrument in [cls.gates, cls.source, cls.meter]:
            instrument.close()

    def test_local(self):
        self.gates.chan1.set(3.33)
        self.assertEqual(self.gates.chan1.get(), 3.33)

        self.gates.reset()
        self.assertEqual(self.gates.chan1.get(), 0)

        with self.assertRaises(ValueError):
            self.gates.ask('knock knock? Oh never mind.')

    def test_instances(self):
        # copied from the main (server-based) version
        # make sure it all works the same here
        instruments = [self.gates, self.source, self.meter]
        for instrument in instruments:
            for other_instrument in instruments:
                instances = instrument.instances()
                # check that each instrument is in only its own
                # instances list
                if other_instrument is instrument:
                    self.assertIn(instrument, instances)
                else:
                    self.assertNotIn(other_instrument, instances)

                # check that we can find each instrument from any other
                # use find_component here to test that it rolls over to
                # find_instrument if only a name is given
                self.assertEqual(
                    instrument,
                    other_instrument.find_component(instrument.name))

                self.assertEqual(
                    instrument.name,
                    other_instrument.find_component(instrument.name + '.name'))

            # check that we can find this instrument from the base class
            self.assertEqual(instrument,
                             Instrument.find_instrument(instrument.name))


class TestModelAttrAccess(TestCase):

    def setUp(self):
        self.model = AMockModel()

    def tearDown(self):
        self.model.close()

    def test_attr_access(self):
        model = self.model

        model.a = 'local'
        with self.assertRaises(AttributeError):
            model.getattr('a')

        self.assertEqual(model.getattr('a', 'dflt'), 'dflt')

        model.setattr('a', 'remote')
        self.assertEqual(model.a, 'local')
        self.assertEqual(model.getattr('a'), 'remote')

        model.delattr('a')
        self.assertEqual(model.getattr('a', 'dflt'), 'dflt')

        model.fmt = 'local override of a remote method'
        self.assertEqual(model.callattr('fmt', 42), '42.000')
        self.assertEqual(model.callattr('fmt', value=12.4), '12.400')


class TestInstrument2(TestCase):

    def setUp(self):
        self.instrument = DummyInstrument(
            name='testdummy', gates=['dac1', 'dac2', 'dac3'], server_name=None)

    def tearDown(self):
        # TODO (giulioungaretti) remove ( does nothing ?)
        pass

    def test_attr_access(self):
        instrument = self.instrument

        # test the instrument works
        instrument.dac1.set(10)
        val = instrument.dac1.get()
        self.assertEqual(val, 10)

        # close the instrument
        instrument.close()

        # make sure we can still print the instrument
        s = instrument.__repr__()

        # make sure the gate is removed
        self.assertEqual(hasattr(instrument, 'dac1'), False)
