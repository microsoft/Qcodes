from typing import Dict, ClassVar
import logging
import time
from bisect import bisect

from qcodes import VisaInstrument, InstrumentChannel, ChannelList
from qcodes.instrument.group_parameter import GroupParameter, Group
from qcodes.utils import validators


log = logging.getLogger(__name__)


class BaseOutput(InstrumentChannel):

    MODES: ClassVar[Dict[str, int]] = {}
    RANGES: ClassVar[Dict[str, int]] = {}

    def __init__(self, parent, output_name, output_index) -> None:
        super().__init__(parent, output_name)
        self.INVERSE_RANGES: Dict[int, str] = {
            v: k for k, v in self.RANGES.items()}
        self._has_pid = True
        self.output_index = output_index

        self.add_parameter('mode',
                           label='Control mode',
                           docstring='Specifies the control mode',
                           val_mapping=self.MODES,
                           parameter_class=GroupParameter)
        self.add_parameter('input_channel',
                           label='Input channel',
                           docstring='Specifies which measurement input to '
                                     'control from (note that only '
                                     'measurement inputs are available)',
                           get_parser=int,
                           parameter_class=GroupParameter)
        self.add_parameter('powerup_enable',
                           label='Power-up enable on/off',
                           docstring='Specifies whether the output remains on '
                                     'or shuts off after power cycle.',
                           val_mapping={True: 1, False: 0},
                           parameter_class=GroupParameter)
        self.output_group = Group([self.mode, self.input_channel,
                                   self.powerup_enable],
                                  set_cmd=f'OUTMODE {output_index}, {{mode}}, '
                                          f'{{input_channel}}, '
                                          f'{{powerup_enable}}',
                                  get_cmd=f'OUTMODE? {output_index}')

        # Parameters for Closed Loop PID Parameter Command
        if self._has_pid:
            self.add_parameter('P',
                               label='P value of closed-loop controller',
                               docstring='The value for control loop '
                                         'Proportional (gain)',
                               vals=validators.Numbers(0, 1000),
                               get_parser=float,
                               parameter_class=GroupParameter)
            self.add_parameter('I',
                               label='I value of closed-loop controller',
                               docstring='The value for control loop '
                                         'Integral (reset)',
                               vals=validators.Numbers(0, 1000),
                               get_parser=float,
                               parameter_class=GroupParameter)
            self.add_parameter('D',
                               label='D value of closed-loop controller',
                               docstring='The value for control loop '
                                         'Derivative (rate)',
                               vals=validators.Numbers(0, 1000),
                               get_parser=float,
                               parameter_class=GroupParameter)
            self.pid_group = Group([self.P, self.I, self.D],
                                   set_cmd=f'PID {output_index}, '
                                           f'{{P}}, {{I}}, {{D}}',
                                   get_cmd=f'PID? {output_index}')

        self.add_parameter('output_range',
                           label='Heater range',
                           docstring='Specifies heater output range. The range '
                                     'setting has no effect if an output is in '
                                     'the `Off` mode, and does not apply to '
                                     'an output in `Monitor Out` mode. '
                                     'An output in `Monitor Out` mode is '
                                     'always on.',
                           val_mapping=self.RANGES,
                           set_cmd=f'RANGE {output_index}, {{}}',
                           get_cmd=f'RANGE? {output_index}')

        self.add_parameter('setpoint',
                           label='Setpoint value (in sensor units)',
                           docstring='The value of the setpoint in the'
                                     'preferred units of the control loop '
                                     'sensor (which is set via '
                                     '`input_channel` parameter)',
                           vals=validators.Numbers(0, 400),
                           get_parser=float,
                           set_cmd=f'SETP {output_index}, {{}}',
                           get_cmd=f'SETP? {output_index}')

        # Additional non-Visa parameters

        self.add_parameter('range_limits',
                           set_cmd=None,
                           vals=validators.Sequence(validators.Numbers(0, 400),
                                                    require_sorted=True,
                                                    length=len(self.RANGES)-1),
                           label='Temperature limits for output ranges',
                           unit='K',
                           docstring='Use this parameter to define which '
                                     'temperature corresponds to which output '
                                     'range; then use the '
                                     '`set_range_from_temperature` method to '
                                     'set the output range via temperature '
                                     'instead of doing it directly')

        self.add_parameter('wait_cycle_time',
                           set_cmd=None,
                           vals=validators.Numbers(0, 100),
                           label='Waiting cycle time',
                           docstring='Time between two readings when waiting '
                                     'for temperature to equilibrate',
                           unit='s')
        self.wait_cycle_time(0.1)

        self.add_parameter('wait_tolerance',
                           set_cmd=None,
                           vals=validators.Numbers(0, 100),
                           label='Waiting tolerance',
                           docstring='Acceptable tolerance when waiting for '
                                     'temperature to equilibrate',
                           unit='')
        self.wait_tolerance(0.1)

        self.add_parameter('wait_equilibration_time',
                           set_cmd=None,
                           vals=validators.Numbers(0, 100),
                           label='Waiting equilibration time',
                           docstring='Duration during which temperature has to '
                                     'be within tolerance',
                           unit='s')
        self.wait_equilibration_time(0.5)

        self.add_parameter('blocking_T',
                           label='Setpoint value with blocking until it is '
                                 'reached',
                           docstring='Sets the setpoint value, and input '
                                     'range, and waits until it is reached. '
                                     'Added for compatibility with Loop.',
                           vals=validators.Numbers(0, 400),
                           get_parser=float,
                           set_cmd=self._set_blocking_T)

    def _set_blocking_T(self, T):
        self.set_range_from_temperature(T)
        self.setpoint(T)
        self.wait_until_set_point_reached()

    def set_range_from_temperature(self, temperature):
        """
        Sets the output range of this given heater from a given temperature.

        The output range is determined by the limits given through the parameter
        `range_limits`. The output range is used for temperatures between
        the limits `range_limits[i-1]` and `range_limits[i]`; that is
        `range_limits` is the upper limit for using a certain heater current.

        Args:
            temperature
                temperature to set the range from

        Returns:
            the value of the resulting `output_range`, that is also available
            from the `output_range` parameter itself
        """
        if self.range_limits.get_latest() is None:
            raise RuntimeError('Error when calling set_range_from_temperature: '
                               'You must specify the output range limits '
                               'before automatically setting the range '
                               '(e.g. inst.range_limits([0.021, 0.1, 0.2, '
                               '1.1, 2, 4, 8]))')
        i = bisect(self.range_limits.get_latest(), temperature)
        # there is a `+1` because `self.RANGES` includes `'off'` as the first
        # value.
        self.output_range(self.INVERSE_RANGES[i+1])
        return self.output_range()

    def set_setpoint_and_range(self, temperature):
        # TODO: Range should be selected according to current temperature,
        # not according to current setpoint
        self.set_range_from_temperature(temperature)
        self.setpoint(temperature)

    def wait_until_set_point_reached(self,
                                     wait_cycle_time: float=None,
                                     wait_tolerance: float=None,
                                     wait_equilibration_time: float=None):
        """
        This function runs a loop that monitors the value of the heater's
        input channel until the read values is close to the setpoint value
        that has been set before.

        Args:
            wait_cycle_time
                this time is being waited between the readings (same as
                `wait_cycle_time` parameter); if None, then the value of the
                corresponding `wait_cycle_time` parameter is used
            wait_tolerance
                this value is used to determine if the reading value is
                close enough to the setpoint value according to the
                following formula:
                `abs(t_reading - t_setpoint)/t_reading < wait_tolerance`
                (same as `wait_tolerance` parameter); if None, then the
                value of the corresponding `wait_tolerance` parameter is used
            wait_equilibration_time:
                within this time, the reading value has to stay within the
                defined tolerance in order for this function to return (same as
                `wait_equilibration_time` parameter); if None, then the value
                of the corresponding `wait_equilibration_time` parameter is used
        """
        wait_cycle_time = wait_cycle_time or self.wait_cycle_time.get_latest()
        wait_tolerance = wait_tolerance or self.wait_tolerance.get_latest()
        wait_equilibration_time = (wait_equilibration_time or
                                   self.wait_equilibration_time.get_latest())

        active_channel_id = self.input_channel()
        active_channel_number_in_list = active_channel_id - 1
        active_channel = self.root_instrument.channels[active_channel_number_in_list]

        if active_channel.units() != 'kelvin':
            raise ValueError(f"Waiting until the setpoint is reached requires "
                             f"channel's {active_channel._channel!r} units to "
                             f"be set to 'kelvin'.")
        
        t_setpoint = self.setpoint()
        
        start_time_in_tolerance_zone = None
        is_in_tolerance_zone = False

        while True:
            t_reading = active_channel.temperature()
            # TODO(MA): if T is lower than sensor range, it keeps on waiting...
            # TODO(DV): only do this coming from one direction
            delta = abs(t_reading - t_setpoint)/t_reading
            log.debug(f'loop iteration with '
                      f't reading of {t_reading}, delta {delta}')

            if delta < wait_tolerance:
                log.debug(f'delta ({delta}) is within '
                          f'wait_tolerance ({wait_tolerance})')
                if is_in_tolerance_zone:
                    if time.monotonic() - start_time_in_tolerance_zone \
                            > wait_equilibration_time:
                        log.debug(f'the reading is within the tolerance zone '
                                  f'for more than wait_equilibration_time '
                                  f'({wait_equilibration_time}), hence exit '
                                  f'the loop')
                        break
                    else:
                        log.debug(f'wait_equilibration_time '
                                  f'({wait_equilibration_time}) within '
                                  f'tolerance zone has not passed yet')
                else:
                    log.debug(f'entering tolerance zone')
                    start_time_in_tolerance_zone = time.monotonic()
                    is_in_tolerance_zone = True
            else:
                log.debug(f'delta ({delta}) is not within '
                          f'wait_tolerance ({wait_tolerance})')
                if is_in_tolerance_zone:
                    log.debug('exiting tolerance zone')
                    is_in_tolerance_zone = False
                    start_time_in_tolerance_zone = None
            
            time.sleep(wait_cycle_time)


class BaseSensorChannel(InstrumentChannel):
    """
    Base class for Lakeshore Temperature Controller sensor channels

    Args:
        parent
            instrument instance that this channel belongs to
        name
            name of the channel
        channel
            string identifier of the channel as referenced in commands;
            for example, '1' or '6' for model 372, or 'A' and 'C' for model 336
    """

    SENSOR_STATUSES: ClassVar[Dict[str, int]] = {}

    def __init__(self, parent, name, channel):
        super().__init__(parent, name)

        self._channel = channel  # Channel on the temperature controller

        # Add the various channel parameters

        self.add_parameter('temperature',
                           get_cmd='KRDG? {}'.format(self._channel),
                           get_parser=float,
                           label='Temperature',
                           unit='K')

        self.add_parameter('t_limit',
                           get_cmd=f'TLIMIT? {self._channel}',
                           set_cmd=f'TLIMIT {self._channel}, {{}}',
                           get_parser=float,
                           label='Temperature limit',
                           docstring='The temperature limit in kelvin for '
                                     'which to shut down all control outputs '
                                     'when exceeded. A temperature limit of '
                                     'zero turns the temperature limit '
                                     'feature off for the given sensor input.',
                           unit='K')

        self.add_parameter('sensor_raw',
                           get_cmd=f'SRDG? {self._channel}',
                           get_parser=float,
                           label='Raw reading',
                           unit='Ohms')  # TODO: This will vary based on sensor type

        self.add_parameter('sensor_status',
                           get_cmd=f'RDGST? {self._channel}',
                           val_mapping=self.SENSOR_STATUSES,
                           label='Sensor status')

        self.add_parameter('sensor_name',
                           get_cmd=f'INNAME? {self._channel}',
                           get_parser=str,
                           set_cmd=f'INNAME {self._channel},\"{{}}\"',
                           vals=validators.Strings(15),
                           label='Sensor name')

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
                           vals=validators.Numbers(1, 200),
                           parameter_class=GroupParameter)
        self.add_parameter('pause',
                           label='Change pause time',
                           docstring='Specifies a value for '
                                     'the change pause time',
                           unit='s',
                           get_parser=int,
                           vals=validators.Numbers(3, 200),
                           parameter_class=GroupParameter)
        self.add_parameter('curve_number',
                           label='Curve',
                           docstring='Specifies which curve the channel uses: '
                                     '0 = no curve, 1 to 59 = standard/user '
                                     'curves. Do not change this parameter '
                                     'unless you know what you are doing.',
                           get_parser=int,
                           vals=validators.Numbers(0, 59),
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
                           vals=validators.Numbers(1, 22),
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


class LakeshoreBase(VisaInstrument):
    """
    This base class has been written to be that base for the Lakeshore 336
    and 372. There are probably other lakeshore modes that can use the
    functionality provided here. If you add another lakeshore driver
    please make sure to extend this class accordingly, or create a new one.
    """
    # Redefine this in the model-specific class in case you want to use a
    # different class for sensor channels
    CHANNEL_CLASS = BaseSensorChannel

    # This dict has channel name in the driver as keys, and channel "name" that
    # is used in instrument commands as values. For example, if channel called
    # "B" is referred to in instrument commands as '2', then this dictionary
    # will contain {'B': '2'} entry.
    channel_name_command: Dict[str, str] = {}

    def __init__(self,
                 name: str,
                 address: str,
                 terminator: str ='\r\n',
                 **kwargs
                 ) -> None:
        super().__init__(name, address, terminator=terminator, **kwargs)

        # Allow access to channels either by referring to the channel name
        # or through a channel list, i.e. instr.A.temperature() and
        # instr.channels[0].temperature() refer to the same parameter.
        self.channels = ChannelList(self, "TempSensors",
                                    self.CHANNEL_CLASS, snapshotable=False)
        for name, command in self.channel_name_command.items():
            channel = self.CHANNEL_CLASS(self, name, command)
            self.channels.append(channel)
            self.add_submodule(name, channel)
        self.channels.lock()
        self.add_submodule("channels", self.channels)

        self.connect_message()
