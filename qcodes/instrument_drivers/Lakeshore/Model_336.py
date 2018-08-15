import os
from typing import ClassVar, Dict

from qcodes.instrument.group_parameter import GroupParameter
from .lakeshore_base import LakeshoreBase, BaseOutput, BaseSensorChannel
import qcodes.utils.validators as vals


# There are 4 sensors channels (a.k.a. measurement inputs) in Model 336.
# Unlike other Lakeshore models, Model 336 refers to the channels using
# letters, and not numbers
_channel_name_to_command_map: Dict[str, str] = {'A': '1',
                                                'B': '2',
                                                'C': '3',
                                                'D': '4'}


class Output_336(BaseOutput):
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

    def __init__(self, parent, output_name, output_index):
        if output_name not in ['A', 'B']:
            self._has_pid = False

        super().__init__(parent, output_name, output_index)

        # Redefine input_channel to use string names instead of numbers
        self.add_parameter('input_channel',
                           label='Input channel',
                           docstring='Specifies which measurement input to '
                                     'control from (note that only '
                                     'measurement inputs are available)',
                           val_mapping=_channel_name_to_command_map,
                           parameter_class=GroupParameter)

        # Add a remark to `mode` parameter docstring
        self.mode.__doc__ += os.linesep.join((
            self.mode.__doc__,
            '',
            'Modes `monitor_out` (4) and `warm_up` (5) are '
            'only valid for Analog Outputs, C (3) and D (4).'
        ))

        self.P.vals = vals.Numbers(0.1, 1000)
        self.I.vals = vals.Numbers(0.1, 1000)
        self.D.vals = vals.Numbers(0, 200)

        self.range_limits.vals = vals.Sequence(
            vals.Numbers(0, 400), length=2, require_sorted=True)


class Model_336_Channel(BaseSensorChannel):
    SENSOR_STATUSES = {'OK': 0,
                       'Invalid Reading': 1,
                       'Temp Underrange': 16,
                       'Temp Overrange': 32,
                       'Sensor Units Zero': 64,
                       'Sensor Units Overrange': 128}

    def __init__(self, parent, name, channel):
        super().__init__(parent, name, channel)


class Model_336(LakeshoreBase):
    """
    Lakeshore Model 336 Temperature Controller Driver
    """
    channel_name_command: Dict[str, str] = _channel_name_to_command_map

    CHANNEL_CLASS = Model_336_Channel

    def __init__(self, name: str, address: str, **kwargs) -> None:
        super().__init__(name, address, **kwargs)

        self.output_1 = Output_336(self, 'output_1', 1)
        self.output_2 = Output_336(self, 'output_2', 2)
