from typing import Dict
from enum import Enum
from collections import OrderedDict
from qcodes.instrument.channel import InstrumentChannel
from qcodes.instrument_drivers.Lakeshore.lakeshore_base import LakeshoreBase, BaseSensorChannel
from qcodes.instrument.parameter import Parameter
from qcodes.utils.command import Command
import qcodes.utils.validators as vals


class GroupParameter(Parameter):
    def __init__(self, name, instrument, **kwargs):
        self.group = None
        super().__init__(name, instrument, **kwargs)

    def set_raw(self, value, **kwargs):
        self.group.set(self, value)

    def get_raw(self, result=None):
        if not result:
            self.group.update()
            return self.raw_value
        else:
            return result


class Group():
    def __init__(self, parameters, set_cmd=None, get_cmd=None,
                 get_parser=None, separator=',', types=None):
        self.parameters = OrderedDict((p.name, p) for p in parameters)
        self.instrument = parameters[0].root_instrument
        for p in parameters:
            p.group = self
        self.set_cmd = set_cmd
        self.get_cmd = get_cmd
        if get_parser:
            self.get_parser = get_parser
        else:
            self.get_parser = self._separator_parser(separator, types)
    
    def _separator_parser(self, separator, types):
        def parser(ret_str):
            print(f'retr string is {ret_str}')
            keys = self.parameters.keys()
            values = ret_str.split(separator)
            return dict(zip(keys, values))
        return parser
    
    def set(self, set_parameter, value):
        if any((p.get_latest() is None) for p in self.parameters.values()):
            self.update()
        calling_dict = {name: p.raw_value for name, p in self.parameters.items()}
        calling_dict[set_parameter.name] = value
        command_str = self.set_cmd.format(**calling_dict)
        set_parameter.root_instrument.write(command_str)

    def update(self):
        ret = self.get_parser(self.instrument.ask(self.get_cmd))
        # if any(p.get_latest() != ret[] for name, p in self.parameters if p is not get_parameter):
        #     log.warn('a value has changed on the device')
        # TODO(DV): this is odd, but the only way to call the wrapper accordingly
        for name, p in list(self.parameters.items()):
            p.get(result=ret[name])
        
        

class Mode(Enum):
    off = 0
    monitor_out = 1
    open_loop = 2
    zone = 3
    still = 4
    closed_loop = 5
    warm_up = 6

class Polarity(Enum):
    unipolar = 0
    bipolar = 1

class Range(Enum):
    off = 0
    max_31_6μA = 1
    max_100μA = 2
    max_316μA = 3
    max_1mA = 4
    max_3_16mA = 5
    max_10mA = 6
    max_31_6mA = 7
    max_100mA = 8

class Model_372_Channel(BaseSensorChannel):
    pass

class Heater(InstrumentChannel):
     def __init__(self, parent, heater_name, heater_index):
        super().__init__(parent, heater_name)
        self.heater_index = heater_index
        self.add_parameter('mode', get_parser=lambda x: Mode(int(x)),
                           vals=vals.Enum(*[i for i in Mode]),
                           set_parser=(lambda enum_value: enum_value.value),
                           parameter_class=GroupParameter)
        self.add_parameter('input_channel', get_parser=int, parameter_class=GroupParameter)
        self.add_parameter('powerup_enable',
                           val_mapping={True: 1, False: 0},
                           parameter_class=GroupParameter)
        self.add_parameter('polarity', get_parser=lambda x: Polarity(int(x)),
                           vals=vals.Enum(*[i for i in Polarity]),
                           set_parser=(lambda enum_value: enum_value.value),
                           parameter_class=GroupParameter)
        self.add_parameter('filter',
                           val_mapping={True: 1, False: 0},
                           parameter_class=GroupParameter)
        self.add_parameter('delay', vals=vals.Ints(0, 255),
                           get_parser=int,
                           parameter_class=GroupParameter)
        self.pid_group = Group([self.mode, self.input_channel,
                               self.powerup_enable, self.polarity,
                               self.filter, self.delay],
                               set_cmd=f"OUTMODE {heater_index}, {{mode}}, {{input_channel}}, {{powerup_enable}}, {{polarity}}, {{filter}}, {{delay}}",
                               get_cmd=f'OUTMODE? {heater_index}')

        self.add_parameter('P', vals=vals.Numbers(0, 1000),
                           get_parser=float, parameter_class=GroupParameter)
        self.add_parameter('I', vals=vals.Numbers(0, 1000),
                           get_parser=float, parameter_class=GroupParameter)
        self.add_parameter('D', vals=vals.Numbers(0, 2500),
                           get_parser=float, parameter_class=GroupParameter)
        self.pid_group = Group([self.P, self.I, self.D],
                               set_cmd=f"PID {heater_index}, {{P}}, {{I}}, {{D}}",
                               get_cmd=f'PID? {heater_index}')
        
        self.add_parameter('range', get_parser=lambda x: Range(int(x)),
                           vals=vals.Enum(*[i for i in Range]),
                           set_parser=(lambda enum_value: enum_value.value),
                           set_cmd=f'RANGE {heater_index}, {{}}',
                           get_cmd=f'RANGE? {heater_index}')

        self.add_parameter('setpoint', get_parser=lambda x: Range(int(x)),
                           vals=vals.Numbers(0, 400),
                           get_parser=float,
                           set_cmd=f'SETP {heater_index}, {{}}',
                           get_cmd=f'SETP? {heater_index}')

class Model_372(LakeshoreBase):
    """
    Lakeshore Model 372 Temperature Controller Driver
    Controlled via sockets
    """
    CHANNEL_CLASS = Model_372_Channel
    channel_name_command: Dict[str, str] = {'ch{:02}'.format(i): str(i) for i in range(1, 17)}
    

    def __init__(self, name: str, address: str, **kwargs) -> None:
        super().__init__(name, address, **kwargs)
        self.sample_heater = Heater(self, 'sample_heater', 0)
        self.warmup_heater = Heater(self, 'warmup_heater', 1)
        self.analog_heater = Heater(self, 'analog_heater', 2)

