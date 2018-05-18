import pytest
import logging
from collections import namedtuple
from functools import wraps
from contextlib import suppress

import qcodes
from qcodes.instrument_drivers.Lakeshore.Model_336 import Model_336
from qcodes.instrument_drivers.Lakeshore.Model_372 import Model_372, Mode, Polarity, Range
import qcodes.instrument.sims as sims


log = logging.getLogger(__name__)


class MockVisaInstrument():

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # This base class mixin holds two dictionaries associated with the 
        # pyvisa_instrument.write()
        self.cmds = {}
        # and pyvisa_instrument.query() functions
        self.queries = {}
        # the keys are the issued VISA commands like '*IDN?' or '*OPC'
        # the values are the corresponding methods to be called on the mock
        # instrument.

        # to facilitate the definition there are the decorators `@query' and `@command`
        # these attach an attribute to the method, so that the dictionaries can be 
        # filled here in the constructor.
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
            log.debug("Calling query on instrument Mock {}: {} for command {} with args {}".format(self.name, cmd, cmd_str, args))
            self.cmds[cmd_str](args)
        else:
            super().write_raw(cmd)

    def ask_raw(self, query):
        query_parts = query.split(' ')
        query_str = query_parts[0].upper()
        if query_str in self.queries:
            args = ''.join(query_parts[1:])
            log.debug("Calling query on instrument Mock {}: {} for command {} with args {}".format(self.name, query, query_str, args))
            print("Calling query on instrument Mock {}: {} for command {} with args {}".format(self.name, query, query_str, args))
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


class Heater:
    def __init__(self, **kwargs):
        for kwarg, value in kwargs.items():
            setattr(self, kwarg, value)

class Model_372_Mock(MockVisaInstrument, Model_372):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.heaters = {}
        # initial values
        self.heaters['0'] = Heater(P=1, I=2, D=3,
                                   mode=5, input_channel=2,
                                   powerup_enable=0, polarity=0,
                                   filter=0, delay=1,
                                   range=0)
        self.heaters['1'] = Heater(P=1, I=2, D=3,
                                   mode=5, input_channel=2,
                                   powerup_enable=0, polarity=0,
                                   filter=0, delay=1,
                                   range=0)
        self.heaters['2'] = Heater(P=1, I=2, D=3,
                                   mode=5, input_channel=2,
                                   powerup_enable=0, polarity=0,
                                   filter=0, delay=1,
                                   range=0)
    @query('PID?')    
    def pidq(self, arg):
        heater = self.heaters[arg]
        return f'{heater.P}, {heater.I}, {heater.D}'

    @command('PID')
    @split_args()
    def pid(self, output, P, I, D):
        self.heaters[output] = Heater(P=P, I=I, D=D)

    @query('OUTMODE?')    
    def outmodeq(self, arg):
        heater = self.heaters[arg]
        return f'{heater.mode}, {heater.input_channel}, {heater.powerup_enable}, {heater.polarity}, {heater.filter}, {heater.delay}'

    @command('OUTMODE')
    @split_args()
    def outputmode(self, output, mode, input_channel, powerup_enable,
                   polarity, filter, delay):
        h = self.heaters[output]
        h.output = output
        h.mode = mode
        h.input_channel = input_channel
        h.powerup_enable = powerup_enable
        h.polarity = polarity
        h.filter = filter
        h.delay = delay
        print(f'setting outputmode to {h.mode}, {input_channel}, {powerup_enable}, {polarity}, {filter}, {delay}')

    @query('RANGE?')    
    def rangeq(self, arg):
        heater = self.heaters[arg]
        return f'{heater.range}'

    @command('RANGE')
    @split_args()
    def heater_range(self, output, heater_range):
        h = self.heaters[output]
        h.range = heater_range
        print(f'setting range to {h.range}')

visalib = sims.__file__.replace('__init__.py', 'lakeshore_model372.yaml@sim')
# def test_instantiation_model_336():
#     ls = Model_336('lakeshore_336', 'GPIB::2::65535::INSTR', visalib=visalib, device_clear=False)
#     ls.close()

# def test_instantiation_model_372():
#     ls = Model_372_Mock('lakeshore_372', 'GPIB::3::65535::INSTR', visalib=visalib, device_clear=False)
#     ls.close()


@pytest.fixture(scope='function')
def lakeshore_372():
    ls = Model_372_Mock('lakeshore_372_fixture', 'GPIB::3::65535::INSTR', visalib=visalib, device_clear=False)
    yield ls
    ls.close()

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
    mode = Mode.closed_loop
    input_channel = 1
    powerup_enable = True
    polarity = Polarity.unipolar
    filter = True
    delay = 1
    # for h in (ls.warmup_heater, ls.analog_heater, ls.sample_heater):
    h = ls.warmup_heater
    h.mode(mode)
    h.input_channel(input_channel)
    h.powerup_enable(powerup_enable)
    h.polarity(polarity)
    h.filter(filter)
    h.delay(delay)
    h.mode()
    assert (h.mode(), h.input_channel()) == (mode, input_channel)
    assert (h.powerup_enable(), h.polarity()) == (powerup_enable, polarity)
    assert (h.filter(), h.delay()) == (filter, delay)

def test_range(lakeshore_372):
    ls = lakeshore_372
    heater_range = Range.max_10mA
    # for h in (ls.warmup_heater, ls.analog_heater, ls.sample_heater):
    h = ls.warmup_heater
    h.range(heater_range)
    assert h.range() == heater_range
