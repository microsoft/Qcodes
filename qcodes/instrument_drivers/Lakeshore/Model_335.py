from typing import ClassVar, Dict, Any

from qcodes.instrument.group_parameter import GroupParameter, Group
from .lakeshore_base import LakeshoreBase, BaseOutput, BaseSensorChannel
from .Lakeshore_model_336 import LakeshoreModel336Channel
import qcodes.utils.validators as vals
import pyvisa as visa


# There are 4 sensors channels (a.k.a. measurement inputs) in Model 336.
# Unlike other Lakeshore models, Model 336 refers to the channels using
# letters, and not numbers
_channel_name_to_command_map: Dict[str, str] = {'A': 'A',
                                                'B': 'B'}

# OUTMODE command of this model refers to the outputs via integer numbers,
# while everywhere else within this model letters are used. This map is
# created in order to preserve uniformity of referencing to sensor channels
# within this driver.
_channel_name_to_outmode_command_map: Dict[str, int] = \
    {ch_name: num_for_cmd + 1
     for num_for_cmd, ch_name in enumerate(_channel_name_to_command_map.keys())}


class LakeshoreModel335CurrentSource(BaseOutput):
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

    def __init__(self, parent, output_name, output_index):
        super().__init__(parent, output_name, output_index, has_pid=True)

        self.P.vals = vals.Numbers(0.1, 1000)
        self.I.vals = vals.Numbers(0.1, 1000)
        self.D.vals = vals.Numbers(0, 200)
        self.setpoint.vals=vals.Numbers(-274, 400) # we want to operate in degC


class LakeshoreModel335(LakeshoreBase):
    """
    Lakeshore Model 335 Temperature Controller Driver
    """
    channel_name_command: Dict[str, str] = _channel_name_to_command_map

    CHANNEL_CLASS = LakeshoreModel336Channel

    input_channel_parameter_values_to_channel_name_on_instrument = \
        _channel_name_to_command_map

    def __init__(self, name: str, address: str, **kwargs) -> None:
        super().__init__(name, address, **kwargs)

        if isinstance(self.visa_handle,visa.resources.serial.SerialInstrument):
            self.visa_handle.baud_rate = 57600
            self.visa_handle.data_bits = 7
            self.visa_handle.parity = visa.constants.Parity(1)

        self.output_1 = LakeshoreModel335CurrentSource(self, 'output_1', 1)
        self.output_2 = LakeshoreModel335CurrentSource(self, 'output_2', 2)
