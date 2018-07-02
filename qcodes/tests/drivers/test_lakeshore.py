import pytest
from typing import Dict, Callable
import logging
from functools import wraps
from contextlib import suppress
import time

# from qcodes.instrument_drivers.Lakeshore.Model_336 import Model_336
from qcodes.instrument_drivers.Lakeshore.Model_372 import Model_372
import qcodes.instrument.sims as sims


log = logging.getLogger(__name__)


class MockVisaInstrument():
    """
    Mixin class that overrides write_raw and ask_raw to simulate an
    instrument.
    """

    def __init__(self, *args, **kwargs) -> None:
        # ignore this line in mypy: Mypy does not support mixins yet
        # and seen by itself with this class definition it does not make sense
        # to call __init__ on the super()
        super().__init__(*args, **kwargs)  # type: ignore
        # This base class mixin holds two dictionaries associated with the
        # pyvisa_instrument.write()
        self.cmds: Dict[str, Callable] = {}
        # and pyvisa_instrument.query() functions
        self.queries: Dict[str, Callable] = {}
        # the keys are the issued VISA commands like '*IDN?' or '*OPC'
        # the values are the corresponding methods to be called on the mock
        # instrument.

        # to facilitate the definition there are the decorators `@query' and
        # `@command`
        # these attach an attribute to the method, so that the dictionaries can
        # be filled here in the constructor.
        # (This is borderline abusive, but makes a it easy to define mocks)
        func_names = dir(self)
        # cycle through all methods
        for func_name in func_names:
            f = getattr(self, func_name)
            # only add for methods that have such an attribue
            with suppress(AttributeError):
                self.queries[getattr(f, 'query_name')] = f
            with suppress(AttributeError):
                self.cmds[getattr(f, 'command_name')] = f


    def write_raw(self, cmd):
        cmd_parts = cmd.split(' ')
        cmd_str = cmd_parts[0].upper()
        if cmd_str in self.cmds:
            args = ''.join(cmd_parts[1:])
            log.debug(f'Calling query on instrument Mock {self.name}: '
                      f'{cmd} for command {cmd_str} with args {args}')
            self.cmds[cmd_str](args)
        else:
            super().write_raw(cmd)

    def ask_raw(self, query):
        query_parts = query.split(' ')
        query_str = query_parts[0].upper()
        if query_str in self.queries:
            args = ''.join(query_parts[1:])
            log.debug(f'Calling query on instrument Mock {self.name}: '
                      f'{query} for command {query_str} with args {args}')
            print(f'Calling query on instrument Mock {self.name}: '
                  f'{query} for command {query_str} with args {args}')
            response = self.queries[query_str](args)
            log.debug(f"Got mock instrument response: {response}")
            return response
        else:
            super().ask_raw(query)


def split_args(split_char=','):
    def wrapper(func):
        @wraps(func)
        def decorated_func(self, string_arg):
            args = string_arg.split(split_char)
            return func(self, *args)
        return decorated_func
    return wrapper


def query(name=None):
    def wrapper(func):
        func.query_name = name.upper()
        return func
    return wrapper


def command(name=None):
    def wrapper(func):
        func.command_name = name.upper()
        return func
    return wrapper


class DictClass:
    def __init__(self, **kwargs):
        for kwarg, value in kwargs.items():
            setattr(self, kwarg, value)


class Model_372_Mock(MockVisaInstrument, Model_372):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.heaters: Dict[str, DictClass] = {}
        # initial values
        self.heaters['0'] = DictClass(P=1, I=2, D=3,
                                      mode=5, input_channel=2,
                                      powerup_enable=0, polarity=0,
                                      use_filter=0, delay=1,
                                      output_range=0,
                                      setpoint=4)
        self.heaters['1'] = DictClass(P=1, I=2, D=3,
                                      mode=5, input_channel=2,
                                      powerup_enable=0, polarity=0,
                                      use_filter=0, delay=1,
                                      output_range=0,
                                      setpoint=4)
        self.heaters['2'] = DictClass(P=1, I=2, D=3,
                                      mode=5, input_channel=2,
                                      powerup_enable=0, polarity=0,
                                      use_filter=0, delay=1,
                                      output_range=0,
                                      setpoint=4)
        self.channel_mock = {str(i): DictClass(tlimit=i, T=4) for i in range(1,17)}

        # simulate delayed heating
        self.simulate_heating = False
        self.start_heating_time = time.perf_counter()

    def start_heating(self):
        self.start_heating_time = time.perf_counter()
        self.simulate_heating = True

    def get_T_when_heating(self):
        """
        Simply define a fixed setpoint of 4 k for now
        """
        delta = abs(time.perf_counter() - self.start_heating_time)
        # make it simple to start with: linear ramp 1k per second
        # start at 7k.
        return max(4, 7 - delta)


    @query('PID?')
    def pidq(self, arg):
        heater = self.heaters[arg]
        return f'{heater.P}, {heater.I}, {heater.D}'

    @command('PID')
    @split_args()
    def pid(self, output, P, I, D):
        for a, v in zip(['P', 'I', 'D'], [P, I, D]):
            setattr(self.heaters[output], a, v)

    @query('OUTMODE?')
    def outmodeq(self, arg):
        heater = self.heaters[arg]
        return (f'{heater.mode}, {heater.input_channel}, '
                f'{heater.powerup_enable}, {heater.polarity}, '
                f'{heater.use_filter}, {heater.delay}')

    @command('OUTMODE')
    @split_args()
    def outputmode(self, output, mode, input_channel, powerup_enable,
                   polarity, use_filter, delay):
        h = self.heaters[output]
        h.output = output
        h.mode = mode
        h.input_channel = input_channel
        h.powerup_enable = powerup_enable
        h.polarity = polarity
        h.use_filter = use_filter
        h.delay = delay
        print(f'setting outputmode to {h.mode}, {input_channel}, '
              f'{powerup_enable}, {polarity}, {use_filter}, {delay}')

    @query('RANGE?')
    def rangeq(self, heater):
        h = self.heaters[heater]
        return f'{h.output_range}'

    @command('RANGE')
    @split_args()
    def range_cmd(self, heater, output_range):
        h = self.heaters[heater]
        h.output_range = output_range
        print(f'setting output_range to {h.output_range}')

    @query('SETP?')
    def setpointq(self, heater):
        h = self.heaters[heater]
        return f'{h.setpoint}'

    @command('SETP')
    @split_args()
    def setpoint(self, heater, setpoint):
        h = self.heaters[heater]
        h.setpoint = setpoint
        print(f'setting setpoint to {h.setpoint}')

    @query('TLIMIT?')
    def tlimitq(self, channel):
        chan = self.channel_mock[channel]
        return f'{chan.tlimit}'

    @command('TLIMIT')
    @split_args()
    def tlimitcmd(self, channel, tlimit):
        chan = self.channel_mock[channel]
        chan.tlimit = tlimit
        print(f'setting TLIMIT to {chan.tlimit}')


    @query('KRDG?')
    def temperature(self, output):
        chan = self.channel_mock[output]
        if self.simulate_heating:
            return self.get_T_when_heating()
        return f'{chan.T}'
# def test_instantiation_model_336():
#     ls = Model_336('lakeshore_336', 'GPIB::2::65535::INSTR', visalib=visalib, device_clear=False)
#     ls.close()

# def test_instantiation_model_372():
#     ls = Model_372_Mock('lakeshore_372', 'GPIB::3::65535::INSTR', visalib=visalib, device_clear=False)
#     ls.close()


def instrument_fixture(scope='function'):
    def wrapper(func):
        @pytest.fixture(scope = scope)
        def wrapped_fixture():
            inst = func()
            try:
                yield inst
            except:
                raise
            finally:
                inst.close()
        return wrapped_fixture
    return wrapper


@instrument_fixture(scope='function')
def lakeshore_372():
    visalib = sims.__file__.replace('__init__.py',
                                    'lakeshore_model372.yaml@sim')
    return Model_372_Mock('lakeshore_372_fixture', 'GPIB::3::65535::INSTR',
                          visalib=visalib, device_clear=False)

def test_pid_set(lakeshore_372):
    ls = lakeshore_372
    P, I, D = 1, 2, 3
    for h in (ls.warmup_heater, ls.analog_heater, ls.sample_heater):
        h.P(P)
        h.I(I)
        h.D(D)
        assert (h.P(), h.I(), h.D()) == (P, I, D)

def test_output_mode(lakeshore_372):
    ls = lakeshore_372
    mode = 'off'
    input_channel = 1
    powerup_enable = True
    polarity = 'unipolar'
    use_filter = True
    delay = 1
    # for h in (ls.warmup_heater, ls.analog_heater, ls.sample_heater):
    h = ls.warmup_heater
    h.mode(mode)
    h.input_channel(input_channel)
    h.powerup_enable(powerup_enable)
    h.polarity(polarity)
    h.use_filter(use_filter)
    h.delay(delay)
    h.mode()
    assert (h.mode(), h.input_channel()) == (mode, input_channel)
    assert (h.powerup_enable(), h.polarity()) == (powerup_enable, polarity)
    assert (h.use_filter(), h.delay()) == (use_filter, delay)

def test_range(lakeshore_372):
    ls = lakeshore_372
    output_range = '10mA'
    # for h in (ls.warmup_heater, ls.analog_heater, ls.sample_heater):
    h = ls.warmup_heater
    h.output_range(output_range)
    assert h.output_range() == output_range

def test_tlimit(lakeshore_372):
    ls = lakeshore_372
    tlimit = 5.1
    # for h in (ls.warmup_heater, ls.analog_heater, ls.sample_heater):
    ls.ch01.t_limit(tlimit)
    assert ls.ch01.t_limit() == tlimit

def test_setpoint(lakeshore_372):
    ls = lakeshore_372
    setpoint = 5.1
    # for h in (ls.warmup_heater, ls.analog_heater, ls.sample_heater):
    ls.sample_heater.setpoint(setpoint)
    assert ls.sample_heater.setpoint() == setpoint

def test_set_and_wait_for_T(lakeshore_372):
    ls = lakeshore_372
    ls.sample_heater.setpoint(4)
    ls.start_heating()
    ls.sample_heater.wait_until_set_point_reached()

def test_select_range_limits(lakeshore_372):
    h = lakeshore_372.sample_heater
    ranges = list(range(1,8))
    h.range_limits(ranges)
    for i in ranges:
        h.set_range_from_temperature(i-0.5)
        assert h.output_range() == h.INVERSE_RANGES[i]

def test_blocking_T(lakeshore_372):
    ls = lakeshore_372
    h = lakeshore_372.sample_heater
    ranges = list(range(1,8))
    h.range_limits(ranges)
    ls.start_heating()
    h.blocking_T(4)
