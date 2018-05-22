from typing import Dict, Union
from enum import Enum
from collections import OrderedDict
from qcodes.instrument.channel import InstrumentChannel
from qcodes.instrument_drivers.Lakeshore.lakeshore_base import LakeshoreBase, BaseSensorChannel, BaseOutput, Group, GroupParameter
from qcodes.utils.command import Command
import qcodes.utils.validators as vals


class Output_372(BaseOutput):
    MODES = {
        'off': 0,
        'monitor_out': 1,
        'open_loop': 2,
        'zone': 3,
        'still': 4,
        'closed_loop': 5,
        'warm_up': 6}
    POLARITIES = {
        'unipolar': 0,
        'bipolar': 1}
    RANGES = {
        'off': 0,
        '31.6μA': 1,
        '100μA': 2,
        '316μA': 3,
        '1mA': 4,
        '3.16mA': 5,
        '10mA': 6,
        '31.6mA': 7,
        '100mA': 8}

    def __init__(self, parent, output_name, output_index):
        super().__init__(parent, output_name, output_index)
        self.add_parameter('polarity',
                           val_mapping=self.POLARITIES,
                           parameter_class=GroupParameter)
        self.add_parameter('filter',
                           val_mapping={True: 1, False: 0},
                           parameter_class=GroupParameter)
        self.add_parameter('delay', vals=vals.Ints(0, 255),
                           get_parser=int,
                           parameter_class=GroupParameter)

        self.output_group = Group([self.mode, self.input_channel,
                                  self.powerup_enable, self.polarity,
                                  self.filter, self.delay],
                                  set_cmd=f"OUTMODE {output_index}, {{mode}}, {{input_channel}}, {{powerup_enable}}, {{polarity}}, {{filter}}, {{delay}}",
                                  get_cmd=f'OUTMODE? {output_index}')




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

