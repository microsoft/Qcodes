from typing import Dict
from collections import OrderedDict
from qcodes.instrument.channel import InstrumentChannel
from qcodes.instrument_drivers.Lakeshore.lakeshore_base import LakeshoreBase, BaseSensorChannel
from qcodes.instrument.parameter import Parameter
from qcodes.utils.command import Command


class GroupParameter(Parameter):
    def __init__(self, name, instrument, **kwargs):
        self.group = None
        super().__init__(name, instrument, **kwargs)

    def set_raw(self, value, **kwargs):
        self.group.set(self, value)

    def get_raw(self, result=None):
        if not result:
            self.group.update()
            return self.get_latest()
        else:
            return result


class Group():
    def __init__(self, parameters, set_cmd=None, get_cmd=None,
                 get_parser=None, separator=','):
        self.parameters = OrderedDict((p.name, p) for p in parameters)
        self.instrument = parameters[0].root_instrument
        for p in parameters:
            p.group = self
        self.set_cmd = set_cmd
        self.get_cmd = get_cmd
        if get_parser:
            self.get_parser = get_parser
        else:
            self.get_parser = self._separator_parser(separator)
    
    def _separator_parser(self, separator):
        def parser(ret_str):
            print(f'retr string is {ret_str}')
            return dict(zip(self.parameters.keys(),
                        (float(arg) for arg in ret_str.split(separator))))
        return parser
    
    def set(self, set_parameter, value):
        if any((p.get_latest() is None) for p in self.parameters.values()):
            self.update()
        calling_dict = {name: p.get_latest() for name, p in self.parameters.items()}
        calling_dict[set_parameter.name] = value
        command_str = self.set_cmd.format(**calling_dict)
        set_parameter.root_instrument.write(command_str)

    def update(self):
        ret = self.get_parser(self.instrument.ask(self.get_cmd))
        # if any(p.get_latest() != ret[] for name, p in self.parameters if p is not get_parameter):
        #     log.warn('a value has changed on the device')
        # TODO(DV): this is odd, but the only way to call the wrapper accordingly
        for name, p in self.parameters.items():
            p.get(result=ret[name])
        
        


class Model_372_Channel(BaseSensorChannel):
    pass

class Heater(InstrumentChannel):
     def __init__(self, parent, heater_name, heater_index):
        super().__init__(parent, heater_name)
        self.heater_index = heater_index
        self.add_parameter('mode', parameter_class=GroupParameter)
        self.add_parameter('input_channel', parameter_class=GroupParameter)
        self.add_parameter('powerup_enable', parameter_class=GroupParameter)
        self.add_parameter('polarity', parameter_class=GroupParameter)
        self.add_parameter('filter', parameter_class=GroupParameter)
        self.add_parameter('delay', parameter_class=GroupParameter)
        self.pid_group = Group([self.mode, self.input_channel,
                               self.powerup_enable, self.polarity,
                               self.filter, self.delay],
                               set_cmd=f"OUTMODE {heater_index}, {{mode}}, {{input_channel}}, {{poweup_enable}}, {{polarity}}, {{filter}}, {{delay}}",
                               get_cmd=f'OUTMODE? {heater_index}')

        self.add_parameter('P', parameter_class=GroupParameter)
        self.add_parameter('I', parameter_class=GroupParameter)
        self.add_parameter('D', parameter_class=GroupParameter)
        self.pid_group = Group([self.P, self.I, self.D],
                               set_cmd=f"PID {heater_index}, {{P}}, {{I}}, {{D}}",
                               get_cmd=f'PID? {heater_index}')
        
        self.add_parameter('range',
                           set_cmd=f'RANGE {heater_index}, {{}}',
                           get_cmd=f'RANGE? {heater_index}')


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

