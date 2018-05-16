import pytest
import logging
from collections import namedtuple

import qcodes
from qcodes.instrument_drivers.Lakeshore.Model_336 import Model_336
from qcodes.instrument_drivers.Lakeshore.Model_372 import Model_372
import qcodes.instrument.sims as sims


log = logging.getLogger(__name__)


class MockVisaInstrument():

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cmds = {}

    def write_raw(self, cmd):
        cmd_parts = cmd.split(' ')
        cmd_str = cmd_parts[0].upper()
        print(f'command string is {cmd_str} and cmd: {cmd}, and len(cmd_parts):{len(cmd_parts)}')
        if cmd_str in self.cmds:
            args = ''.join(cmd_parts[1:])
            log.debug("Calling query on instrument Mock {}: {} for command {} with args {}".format(self.name, cmd, cmd_str, args))
            print("Calling write on instrument Mock {}: {} for command {} with args {}".format(self.name, cmd, cmd_str, args))
            self.cmds[cmd_str](args)
        else:
            super().write_raw(cmd)

    def ask_raw(self, cmd):
        cmd_parts = cmd.split(' ')
        cmd_str = cmd_parts[0].upper()
        print(f'command string is {cmd_str} and cmd: {cmd}, and len(cmd_parts):{len(cmd_parts)}')
        if cmd_str in self.cmds:
            args = ''.join(cmd_parts[1:])
            log.debug("Calling query on instrument Mock {}: {} for command {} with args {}".format(self.name, cmd, cmd_str, args))
            print("Calling query on instrument Mock {}: {} for command {} with args {}".format(self.name, cmd, cmd_str, args))
            response = self.cmds[cmd_str](args)
            log.debug(f"Got mock instrument response: {response}")
            return response
        else:
            super().ask_raw(cmd)

def split_args(split_char=','):
    def decorator(func):
        def decorated_func(string_arg):
            args = string_arg.split(split_char)
            return func(args)
        return decorated_func
    return decorator

Heater = namedtuple('Heater', ('P', 'I', 'D'))    


class Model_372_Mock(MockVisaInstrument, Model_372):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cmds['PID?'] = self.pidq
        self.cmds['PID'] = self.pid
        self.heaters = {}
        self.heaters['0'] = Heater(P=1, I=2, D=3)
    
    def pidq(self, arg):
        print('reached pidq')
        heater = self.heaters[arg]
        return f'{heater.P}, {heater.I}, {heater.D}'

    @split_args()
    def pid(self, output, P, I, D):
        heater = self.heaters[output]
        heater.P = P
        heater.I = I
        heater.D = D
        print('reached pids')

def test_instantiation_model_336():
    visalib = sims.__file__.replace('__init__.py', 'lakeshore_model336.yaml@sim')
    ls = Model_336('lakeshore_336', 'GPIB::2::65535::INSTR', visalib=visalib, device_clear=False)


def test_instantiation_model_372():
    visalib = sims.__file__.replace('__init__.py', 'lakeshore_model372.yaml@sim')
    ls = Model_372_Mock('lakeshore_372', 'GPIB::3::65535::INSTR', visalib=visalib, device_clear=False)
    ls.P(1)
