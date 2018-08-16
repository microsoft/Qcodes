from typing import Dict, ClassVar

from qcodes.instrument_drivers.Lakeshore.lakeshore_base import (
    LakeshoreBase, BaseOutput, BaseSensorChannel)
from qcodes.instrument.group_parameter import GroupParameter, Group
import qcodes.utils.validators as vals


# There are 16 sensors channels (a.k.a. measurement inputs) in Model 372
_n_channels = 16


class Output_372(BaseOutput):
    """Class for control outputs (heaters) of model 372"""

    MODES: ClassVar[Dict[str, int]] = {
        'off': 0,
        'monitor_out': 1,
        'open_loop': 2,
        'zone': 3,
        'still': 4,
        'closed_loop': 5,
        'warm_up': 6}
    POLARITIES: ClassVar[Dict[str, int]] = {
        'unipolar': 0,
        'bipolar': 1}
    RANGES: ClassVar[Dict[str, int]] = {
        'off': 0,
        '31.6μA': 1,
        '100μA': 2,
        '316μA': 3,
        '1mA': 4,
        '3.16mA': 5,
        '10mA': 6,
        '31.6mA': 7,
        '100mA': 8}

    def __init__(self, parent, output_name, output_index, **kwargs) -> None:
        super().__init__(parent, output_name, output_index, **kwargs)

        self.input_channel.vals = vals.Numbers(1, _n_channels)

        # Add more parameters for OUTMODE command 
        # and redefine the corresponding group
        self.add_parameter('polarity',
                           label='Output polarity',
                           docstring='Specifies output polarity (not '
                                     'applicable to warm-up heater)',
                           val_mapping=self.POLARITIES,
                           parameter_class=GroupParameter)
        self.add_parameter('use_filter',
                           label='Use filter for readings',
                           docstring='Specifies controlling on unfiltered or '
                                     'filtered readings',
                           val_mapping={True: 1, False: 0},
                           parameter_class=GroupParameter)
        self.add_parameter('delay',
                           label='Delay',
                           unit='s',
                           docstring='Delay in seconds for setpoint change '
                                     'during Autoscanning',
                           vals=vals.Ints(0, 255),
                           get_parser=int,
                           parameter_class=GroupParameter)
        self.output_group = Group([self.mode, self.input_channel,
                                   self.powerup_enable, self.polarity,
                                   self.use_filter, self.delay],
                                  set_cmd=f'OUTMODE {output_index}, {{mode}}, '
                                          f'{{input_channel}}, '
                                          f'{{powerup_enable}}, {{polarity}}, '
                                          f'{{use_filter}}, {{delay}}',
                                  get_cmd=f'OUTMODE? {output_index}')

        self.P.vals = vals.Numbers(0.0, 1000)
        self.I.vals = vals.Numbers(0.0, 10000)
        self.D.vals = vals.Numbers(0, 2500)


class Model_372_Channel(BaseSensorChannel):
    SENSOR_STATUSES = {0: 'OK',
                       1: 'CS OVL',
                       2: 'VCM OVL',
                       4: 'VMIX OVL',
                       8: 'VDIF OVL',
                       16: 'R. OVER',
                       32: 'R. UNDER',
                       64: 'T. OVER',
                       128: 'T. UNDER'}

    def __init__(self, parent, name, channel):
        super().__init__(parent, name, channel)

        # Parameters related to Input Channel Parameter Command (INSET)
        self.add_parameter('enabled',
                           label='Enabled',
                           docstring='Specifies whether the input/channel is '
                                     'enabled or disabled. At least one '
                                     'measurement input channel must be '
                                     'enabled. If all are configured to '
                                     'disabled, channel 1 will change to '
                                     'enabled.',
                           val_mapping={True: 1, False: 0},
                           parameter_class=GroupParameter)
        self.add_parameter('dwell',
                           label='Dwell',
                           docstring='Specifies a value for the autoscanning '
                                     'dwell time.',
                           unit='s',
                           get_parser=int,
                           vals=vals.Numbers(1, 200),
                           parameter_class=GroupParameter)
        self.add_parameter('pause',
                           label='Change pause time',
                           docstring='Specifies a value for '
                                     'the change pause time',
                           unit='s',
                           get_parser=int,
                           vals=vals.Numbers(3, 200),
                           parameter_class=GroupParameter)
        self.add_parameter('curve_number',
                           label='Curve',
                           docstring='Specifies which curve the channel uses: '
                                     '0 = no curve, 1 to 59 = standard/user '
                                     'curves. Do not change this parameter '
                                     'unless you know what you are doing.',
                           get_parser=int,
                           vals=vals.Numbers(0, 59),
                           parameter_class=GroupParameter)
        self.add_parameter('temperature_coefficient',
                           label='Change pause time',
                           docstring='Sets the temperature coefficient that '
                                     'will be used for temperature control if '
                                     'no curve is selected (negative or '
                                     'positive). Do not change this parameter '
                                     'unless you know what you are doing.',
                           val_mapping={'negative': 1, 'positive': 2},
                           parameter_class=GroupParameter)
        self.output_group = Group([self.enabled, self.dwell, self.pause,
                                   self.curve_number,
                                   self.temperature_coefficient],
                                  set_cmd=f'INSET {self._channel}, '
                                          f'{{enabled}}, {{dwell}}, {{pause}}, '
                                          f'{{curve_number}}, '
                                          f'{{temperature_coefficient}}',
                                  get_cmd=f'INSET? {self._channel}')


class Model_372(LakeshoreBase):
    """
    Lakeshore Model 372 Temperature Controller Driver

    Note that interaction with the control input (referred to as 'A' in the 
    Computer Interface Operation section of the manual) is not implemented.
    """
    channel_name_command: Dict[str, str] = {'ch{:02}'.format(i): str(i)
                                            for i in range(1, 1 + _n_channels)}

    CHANNEL_CLASS = Model_372_Channel

    def __init__(self, name: str, address: str, **kwargs) -> None:
        super().__init__(name, address, **kwargs)
        self.sample_heater = Output_372(self, 'sample_heater', 0)
        self.warmup_heater = Output_372(self, 'warmup_heater', 1)
        self.analog_heater = Output_372(self, 'analog_heater', 2)
