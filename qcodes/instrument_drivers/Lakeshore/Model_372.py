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
    def __init__(self, parameters, set_cmd=None, get_cmd=None, get_parser=None, separator=','):
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

# class Heater(InstrumentChannel):
#      def __init__(self, parent, heater_name, heater_index):
#         super().__init__(parent, heater_name)
#         self.heater_index = heater_index
#         self.add_parameter('input_channel', type=GroupParameter, group=group)
#         self.add_parameter('', set_cmd=self._set_input, get_cmd=self._get_input)

#      def _set_input(self, value, **kwargs):
         

#      def _get_input(self):
#          pass

class Model_372(LakeshoreBase):
    """
    Lakeshore Model 372 Temperature Controller Driver
    Controlled via sockets
    """
    CHANNEL_CLASS = Model_372_Channel
    channel_name_command: Dict[str,str] = {'ch{:02}'.format(i): str(i) for i in range(1,17)}
    def __init__(self, name: str, address: str, **kwargs) -> None:
        super().__init__(name, address, **kwargs)

        self.add_parameter('P', parameter_class=GroupParameter)
        self.add_parameter('I', parameter_class=GroupParameter)
        self.add_parameter('D', parameter_class=GroupParameter)
        self.pid_group = Group([self.P, self.I, self.D],
                               set_cmd="PID 1, {P}, {I}, {D}",
                               get_cmd='PID? 1')



# class Model_372(LakeshoreBase):
#     """
#     Lakeshore Model 372 Temperature Controller Driver
#     Controlled via sockets
#     """
#     CHANNEL_CLASS = Model_372_Channel
#     channel_name_command: Dict[str,str] = {'ch{:02}'.format(i): str(i) for i in range(1,17)}
#     def __init__(self, name: str, address: str, **kwargs) -> None:
#         super().__init__(name, address, **kwargs)

#         self.add_parameter('P', set_cmd=self._set_p, get_cmd=self._get_p)
#         self.add_parameter('I', set_cmd=self._set_i, get_cmd=self._get_i)
#         self.add_parameter('D', set_cmd=self._set_d, get_cmd=self._get_d)

#         # self.add_parameter('selected_heater', set_cmd=None, val_mapping={'SAMPLE_HEATER': 0, 'OUTPUT_HEATER': 1})
#         self.add_parameter('output', set_cmd=None, val_mapping={'SAMPLE_HEATER': 0, 'OUTPUT_HEATER': 1})


#     def selected_heater(self):
#         return '0'

#     def _get_pid(self):
#         return [float(num) for num in self.ask(f"PID? {self.selected_heater()}").split(',')]

#     def _set_p(self, P):
#         self.write(f"PID {self.selected_heater()}, {P}, {self.I()}, {self.D()}")

#     def _get_p(self):
#         return self._get_pid()[0]

#     def _set_i(self, i):
#         self.write(f"PID {self.selected_heater()}, {self.P()}, {i}, {self.D()}")

#     def _get_i(self):
#         return self._get_pid()[1]

#     def _set_d(self, d):
#         self.write(f"PID {self.selected_heater()}, {self.P()}, {self.I()}, {d}")

#     def _get_d(self):
#         return self._get_pid()[2]
