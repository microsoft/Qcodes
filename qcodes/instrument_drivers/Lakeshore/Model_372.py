from typing import Dict
from qcodes.instrument.channel import InstrumentChannel
from qcodes.instrument_drivers.Lakeshore.lakeshore_base import LakeshoreBase, BaseSensorChannel
from qcodes.instrument.parameter import Parameter
from qcodes.utils.command import Command


# class GroupParameter(Parameter):
#     def __init__(self, name, instrument, *, group=None, **kwargs):
#         self.group = group
#         self.group.add(self)
#         super().__init__(name, instrument, **kwargs)

#     def set_raw(self, value, **kwargs):
#         self.group.update()

#     def get_raw(self):
#         return self.group.get(self.name)


# class Group():
#     def __init__(self, set_cmd=None, get_cmd=None):
#         self.parameters = []

#     def add(self, parameter: GroupParameter)
#         # assert they are all of the same instrument
#         self.parameters.append(parameter)
    
#     def update(self):
#         calling_dict = {p.name: p.get_latest() for p in self.parameters}
#         command_str = self.instrument.set_cmd.format()
#         self.set_cmd(wrapping)


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

        self.add_parameter('P', set_cmd=self._set_p, get_cmd=self._get_p)
        self.add_parameter('I', set_cmd=self._set_i, get_cmd=self._get_i)
        self.add_parameter('D', set_cmd=self._set_d, get_cmd=self._get_d)

        # self.add_parameter('selected_heater', set_cmd=None, val_mapping={'SAMPLE_HEATER': 0, 'OUTPUT_HEATER': 1})
        self.add_parameter('output', set_cmd=None, val_mapping={'SAMPLE_HEATER': 0, 'OUTPUT_HEATER': 1})


    def selected_heater(self):
        return '0'

    def _get_pid(self):
        return [float(num) for num in self.ask(f"PID?").split(',')]

    def _set_p(self, P):
        self.write(f"PID {self.selected_heater()}, {P}, {self.I()}, {self.D()}")

    def _get_p(self):
        return self._get_pid()[0]

    def _set_i(self, i):
        self.write(f"PID {self.selected_heater()}, {self.P()}, {i}, {self.D()}")

    def _get_i(self):
        return self._get_pid()[1]

    def _set_d(self, d):
        self.write(f"PID {self.selected_heater()}, {self.P()}, {self.I()}, {d}")

    def _get_d(self):
        return self._get_pid()[2]
