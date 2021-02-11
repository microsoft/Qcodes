from typing import ClassVar, Dict, Any

from qcodes.instrument.group_parameter import GroupParameter, Group
from .lakeshore_base import LakeshoreBase, BaseOutput, BaseSensorChannel
import qcodes.utils.validators as vals


# There are 4 sensors channels (a.k.a. measurement inputs) in Model 336.
# Unlike other Lakeshore models, Model 336 refers to the channels using
# letters, and not numbers
_channel_name_to_command_map: Dict[str, str] = {'A': 'A',
                                                'B': 'B',
                                                'C': 'C',
                                                'D': 'D'}

# OUTMODE command of this model refers to the outputs via integer numbers,
# while everywhere else within this model letters are used. This map is
# created in order to preserve uniformity of referencing to sensor channels
# within this driver.
_channel_name_to_outmode_command_map: Dict[str, int] = \
    {ch_name: num_for_cmd + 1
     for num_for_cmd, ch_name in enumerate(_channel_name_to_command_map.keys())}


class Output_336_CurrentSource(BaseOutput):
    """
    Class for control outputs 1 and 2 of model 336 that are variable DC current
    sources referenced to chassis ground
    """

    MODES: ClassVar[Dict[str, int]] = {
        'off': 0,
        'closed_loop': 1,
        'zone': 2,
        'open_loop': 3}

    RANGES: ClassVar[Dict[str, int]] = {
        'off': 0,
        'low': 1,
        'medium': 2,
        'high': 3}

    _input_channel_parameter_kwargs = {
        'val_mapping': _channel_name_to_outmode_command_map}

    def __init__(
            self,
            parent: "Model_336",
            output_name: str,
            output_index: int
    ):
        super().__init__(parent, output_name, output_index, has_pid=True)

        self.P.vals = vals.Numbers(0.1, 1000)
        self.I.vals = vals.Numbers(0.1, 1000)
        self.D.vals = vals.Numbers(0, 200)


class Output_336_VoltageSource(BaseOutput):
    """
    Class for control outputs 3 and 4 of model 336 that are variable DC voltage
    sources
    """

    MODES: ClassVar[Dict[str, int]] = {
        'off': 0,
        'closed_loop': 1,
        'zone': 2,
        'open_loop': 3,
        'monitor_out': 4,
        'warm_up': 5}

    RANGES: ClassVar[Dict[str, int]] = {
        'off': 0,
        'low': 1,
        'medium': 2,
        'high': 3}

    def __init__(
            self,
            parent: "Model_336",
            output_name: str,
            output_index: int
    ):
        super().__init__(parent, output_name, output_index, has_pid=False)


class Model_336_Channel(BaseSensorChannel):
    SENSOR_STATUSES = {0: 'OK',
                       1: 'Invalid Reading',
                       16: 'Temp Underrange',
                       32: 'Temp Overrange',
                       64: 'Sensor Units Zero',
                       128: 'Sensor Units Overrange'}

    def __init__(
            self,
            parent: "Model_336",
            name: str,
            channel: str
    ):
        super().__init__(parent, name, channel)

        # Parameters related to Input Type Parameter Command (INTYPE)
        self.add_parameter('sensor_type',
                           label='Input sensor type',
                           docstring='Specifies input sensor type',
                           val_mapping={'disabled': 0,
                                        'diode': 1,
                                        'platinum_rtd': 2,
                                        'ntc_rtd': 3},
                           parameter_class=GroupParameter)
        self.add_parameter('auto_range_enabled',
                           label='Autoranging',
                           docstring='Specifies if autoranging is enabled. '
                                     'Does not apply for diode sensor type',
                           val_mapping={False: 0, True: 1},
                           parameter_class=GroupParameter)
        self.add_parameter('range',
                           label='Range',
                           docstring='Specifies input range when autorange is '
                                     'not enabled. If autorange is on, the '
                                     'returned value corresponds to the '
                                     'currently auto-selected range. The list '
                                     'of available ranges depends on the '
                                     'chosen sensor type: diode 0-1, platinum '
                                     'RTD 0-6, NTC RTD 0-8. Refer to the page '
                                     '136 of the manual for the lookup table',
                           vals=vals.Numbers(0, 8),
                           parameter_class=GroupParameter)
        self.add_parameter('compensation_enabled',
                           label='Compensation enabled',
                           docstring='Specifies input compensation. Reversal '
                                     'for thermal EMF compensation if input '
                                     'is resistive, room compensation if '
                                     'input is thermocouple. Always 0 if input '
                                     'is a diode',
                           val_mapping={False: 0, True: 1},
                           parameter_class=GroupParameter)
        self.add_parameter('units',
                           label='Preferred units',
                           docstring='Specifies the preferred units parameter '
                                     'for sensor readings and for the control '
                                     'setpoint (kelvin, celsius, or sensor)',
                           val_mapping={'kelvin': 1, 'celsius': 2, 'sensor': 3},
                           parameter_class=GroupParameter)
        self.output_group = Group([self.sensor_type, self.auto_range_enabled,
                                   self.range, self.compensation_enabled,
                                   self.units],
                                  set_cmd=f'INTYPE {self._channel}, '
                                          f'{{sensor_type}}, '
                                          f'{{auto_range_enabled}}, {{range}}, '
                                          f'{{compensation_enabled}}, '
                                          f'{{units}}',
                                  get_cmd=f'INTYPE? {self._channel}')


class Model_336(LakeshoreBase):
    """
    Lakeshore Model 336 Temperature Controller Driver
    """
    channel_name_command: Dict[str, str] = _channel_name_to_command_map

    CHANNEL_CLASS = Model_336_Channel

    input_channel_parameter_values_to_channel_name_on_instrument = \
        _channel_name_to_command_map

    def __init__(self, name: str, address: str, **kwargs: Any) -> None:
        super().__init__(name, address, **kwargs)

        self.output_1 = Output_336_CurrentSource(self, 'output_1', 1)
        self.output_2 = Output_336_CurrentSource(self, 'output_2', 2)
        self.output_3 = Output_336_VoltageSource(self, 'output_3', 3)
        self.output_4 = Output_336_VoltageSource(self, 'output_4', 4)
