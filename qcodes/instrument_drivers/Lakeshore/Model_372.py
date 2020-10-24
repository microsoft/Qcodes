from typing import Dict, ClassVar, Any

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

    _input_channel_parameter_kwargs: ClassVar[Dict[str, Any]] = {
        'get_parser': int,
        'vals': vals.Numbers(1, _n_channels)}

    def __init__(
            self,
            parent: "Model_372",
            output_name: str,
            output_index: int
    ) -> None:
        super().__init__(parent, output_name, output_index, has_pid=True)

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

    def __init__(
            self,
            parent: "Model_372",
            name: str,
            channel: str
    ):
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

        # Parameters related to Input Setup Command (INTYPE)
        self.add_parameter('excitation_mode',
                           label='Excitation mode',
                           docstring='Specifies excitation mode',
                           val_mapping={'voltage': 0, 'current': 1},
                           parameter_class=GroupParameter)
        # The allowed values for this parameter change based on the value of
        # the 'excitation_mode' parameter. Moreover, there is a table in the
        # manual that assigns the numbers to particular voltage/current ranges.
        # Once this parameter is heavily used, it can be implemented properly
        # (i.e. using val_mapping, and that val_mapping is updated based on the
        # value of 'excitation_mode'). At the moment, this parameter is added
        # only because it is a part of a group.
        self.add_parameter('excitation_range_number',
                           label='Excitation range number',
                           docstring='Specifies excitation range number '
                                     '(1-12 for voltage excitation, 1-22 for '
                                     'current excitation); refer to the manual '
                                     'for the table of ranges',
                           get_parser=int,
                           vals=vals.Numbers(1, 22),
                           parameter_class=GroupParameter)
        self.add_parameter('auto_range',
                           label='Auto range',
                           docstring='Specifies auto range setting',
                           val_mapping={'off': 0, 'current': 1},
                           parameter_class=GroupParameter)
        self.add_parameter('range',
                           label='Range',
                           val_mapping={'2.0 mOhm': 1,
                                        '6.32 mOhm': 2,
                                        '20.0 mOhm': 3,
                                        '63.2 mOhm': 4,
                                        '200 mOhm': 5,
                                        '632 mOhm': 6,
                                        '2.00 Ohm': 7,
                                        '6.32 Ohm': 8,
                                        '20.0 Ohm': 9,
                                        '63.2 Ohm': 10,
                                        '200 Ohm': 11,
                                        '632 Ohm': 12,
                                        '2.00 kOhm': 13,
                                        '6.32 kOhm': 14,
                                        '20.0 kOhm': 15,
                                        '63.2 kOhm': 16,
                                        '200 kOhm': 17,
                                        '632 kOhm': 18,
                                        '2.0 MOhm': 19,
                                        '6.32 MOhm': 20,
                                        '20.0 MOhm': 21,
                                        '63.2 MOhm': 22},
                           parameter_class=GroupParameter)
        self.add_parameter('current_source_shunted',
                           label='Current source shunt',
                           docstring='Current source either not shunted '
                                     '(excitation on), or shunted '
                                     '(excitation off)',
                           val_mapping={False: 0, True: 1},
                           parameter_class=GroupParameter)
        self.add_parameter('units',
                           label='Preferred units',
                           docstring='Specifies the preferred units parameter '
                                     'for sensor readings and for the control '
                                     'setpoint (kelvin or ohms)',
                           val_mapping={'kelvin': 1, 'ohms': 2},
                           parameter_class=GroupParameter)
        self.output_group = Group([self.excitation_mode,
                                   self.excitation_range_number,
                                   self.auto_range, self.range,
                                   self.current_source_shunted, self.units],
                                  set_cmd=f'INTYPE {self._channel}, '
                                          f'{{excitation_mode}}, '
                                          f'{{excitation_range_number}}, '
                                          f'{{auto_range}}, {{range}}, '
                                          f'{{current_source_shunted}}, '
                                          f'{{units}}',
                                  get_cmd=f'INTYPE? {self._channel}')


class Model_372(LakeshoreBase):
    """
    Lakeshore Model 372 Temperature Controller Driver

    Note that interaction with the control input (referred to as 'A' in the
    Computer Interface Operation section of the manual) is not implemented.
    """
    channel_name_command: Dict[str, str] = {f'ch{i:02}': str(i)
                                            for i in range(1, 1 + _n_channels)}
    input_channel_parameter_values_to_channel_name_on_instrument = {
        i: f'ch{i:02}' for i in range(1, 1 + _n_channels)
    }

    CHANNEL_CLASS = Model_372_Channel

    def __init__(self, name: str, address: str, **kwargs: Any) -> None:
        super().__init__(name, address, **kwargs)

        heaters = {'sample_heater': 0, 'warmup_heater': 1, 'analog_heater': 2}
        for heater_name, heater_index in heaters.items():
            self.add_submodule(heater_name, Output_372(self, heater_name, heater_index))
