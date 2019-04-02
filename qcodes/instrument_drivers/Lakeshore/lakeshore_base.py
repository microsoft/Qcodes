from typing import Dict, ClassVar, List, Any
import time
from bisect import bisect

import numpy as np

from qcodes import VisaInstrument, InstrumentChannel, ChannelList
from qcodes.instrument.group_parameter import GroupParameter, Group
from qcodes.utils import validators as vals


class BaseOutput(InstrumentChannel):
    """
    Base class for the outputs of Lakeshore temperature controllers

    Args:
        parent
            instrument that this channel belongs to
        output_name
            name of this output
        output_index
            identifier for this output that is used in VISA commands of the
            instrument
        has_pid
            if True, then the output supports closed loop control,
            hence it will have three parameters to set it up: 'P', 'I', and 'D'
    """

    MODES: ClassVar[Dict[str, int]] = {}
    RANGES: ClassVar[Dict[str, int]] = {}

    _input_channel_parameter_kwargs: ClassVar[Dict[str, Any]] = {}

    def __init__(self, parent, output_name, output_index, has_pid: bool=True) \
            -> None:
        super().__init__(parent, output_name)

        self.INVERSE_RANGES: Dict[int, str] = {
            v: k for k, v in self.RANGES.items()}

        self._has_pid = has_pid
        self._output_index = output_index

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
                           parameter_class=GroupParameter,
                           **self._input_channel_parameter_kwargs)
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
                               label='Proportional (closed-loop)',
                               docstring='The value for closed control loop '
                                         'Proportional (gain)',
                               vals=vals.Numbers(0, 1000),
                               get_parser=float,
                               parameter_class=GroupParameter)
            self.add_parameter('I',
                               label='Integral (closed-loop)',
                               docstring='The value for closed control loop '
                                         'Integral (reset)',
                               vals=vals.Numbers(0, 1000),
                               get_parser=float,
                               parameter_class=GroupParameter)
            self.add_parameter('D',
                               label='Derivative (closed-loop)',
                               docstring='The value for closed control loop '
                                         'Derivative (rate)',
                               vals=vals.Numbers(0, 1000),
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
                           docstring='The value of the setpoint in the '
                                     'preferred units of the control loop '
                                     'sensor (which is set via '
                                     '`input_channel` parameter)',
                           vals=vals.Numbers(0, 400),
                           get_parser=float,
                           set_cmd=f'SETP {output_index}, {{}}',
                           get_cmd=f'SETP? {output_index}')

        # Additional non-Visa parameters

        self.add_parameter('range_limits',
                           set_cmd=None,
                           get_cmd=None,
                           vals=vals.Sequence(vals.Numbers(0, 400),
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
                           get_cmd=None,
                           vals=vals.Numbers(0, 100),
                           label='Waiting cycle time',
                           docstring='Time between two readings when waiting '
                                     'for temperature to equilibrate',
                           unit='s')
        self.wait_cycle_time(0.1)

        self.add_parameter('wait_tolerance',
                           set_cmd=None,
                           get_cmd=None,
                           vals=vals.Numbers(0, 100),
                           label='Waiting tolerance',
                           docstring='Acceptable tolerance when waiting for '
                                     'temperature to equilibrate',
                           unit='')
        self.wait_tolerance(0.1)

        self.add_parameter('wait_equilibration_time',
                           set_cmd=None,
                           get_cmd=None,
                           vals=vals.Numbers(0, 100),
                           label='Waiting equilibration time',
                           docstring='Duration during which temperature has to '
                                     'be within tolerance',
                           unit='s')
        self.wait_equilibration_time(0.5)

        self.add_parameter('blocking_t',
                           label='Setpoint value with blocking until it is '
                                 'reached',
                           docstring='Sets the setpoint value, and input '
                                     'range, and waits until it is reached. '
                                     'Added for compatibility with Loop. Note '
                                     'that if the setpoint value is in '
                                     'a different range, this function may '
                                     'wait forever because that setpoint '
                                     'cannot be reached within the current '
                                     'range.',
                           vals=vals.Numbers(0, 400),
                           get_parser=float,
                           set_cmd=self._set_blocking_t)

    def _set_blocking_t(self, temperature):
        self.set_range_from_temperature(temperature)
        self.setpoint(temperature)
        self.wait_until_set_point_reached()

    def set_range_from_temperature(self, temperature: float):
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
        range_limits = self.range_limits.get_latest()
        i = bisect(range_limits, temperature)
        # if temperature is larger than the highest range, then bisect returns
        # an index that is +1 from the needed index, hence we need to take
        # care of this corner case here:
        i = min(i, len(range_limits) - 1)
        # there is a `+1` because `self.RANGES` includes `'off'` as the first
        # value.
        orange = self.INVERSE_RANGES[i+1] # this is `output range` not the fruit
        self.log.debug(f'setting output range from temperature '
                       f'({temperature} K) to {orange}.')
        self.output_range(orange)
        return self.output_range()

    def set_setpoint_and_range(self, temperature: float):
        """
        Sets the range from the given temperature, and then sets the setpoint
        to this given temperature.

        Note that the preferred units of the heater output are expected to be
        kelvin.

        Args:
            temperature
                temperature in K
        """
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

        Note that if the setpoint value is in a different range,
        this function may wait forever because that setpoint cannot be
        reached within the current range.

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
        tolerance = wait_tolerance or self.wait_tolerance.get_latest()
        equilibration_time = (wait_equilibration_time or
                                   self.wait_equilibration_time.get_latest())

        active_channel_id = self.input_channel()
        active_channel_name_on_instrument = (self.root_instrument
            .input_channel_parameter_values_to_channel_name_on_instrument[active_channel_id])
        active_channel = getattr(self.root_instrument, active_channel_name_on_instrument)

        if active_channel.units() != 'kelvin':
            raise ValueError(f"Waiting until the setpoint is reached requires "
                             f"channel's {active_channel._channel!r} units to "
                             f"be set to 'kelvin'.")

        t_setpoint = self.setpoint()

        time_now = time.perf_counter()
        time_enter_tolerance_zone = time_now

        while time_now - time_enter_tolerance_zone < equilibration_time:
            time_now = time.perf_counter()

            t_reading = active_channel.temperature()

            if abs(t_reading - t_setpoint) / t_reading > tolerance:
                # Reset time_enter_tolerance_zone to time_now because we left
                # the tolerance zone here (if we even were inside one)
                time_enter_tolerance_zone = time_now

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

    # A dictionary of sensor statuses that assigns a string representation of
    # the status to a status bit weighting (e.g. {4: 'VMIX OVL'})
    SENSOR_STATUSES: ClassVar[Dict[int, str]] = {}

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
                           unit='Ohms')

        self.add_parameter('sensor_status',
                           get_cmd=f'RDGST? {self._channel}',
                           get_parser=self._decode_sensor_status,
                           label='Sensor status')

        self.add_parameter('sensor_name',
                           get_cmd=f'INNAME? {self._channel}',
                           get_parser=str,
                           set_cmd=f'INNAME {self._channel},\"{{}}\"',
                           vals=vals.Strings(15),
                           label='Sensor name')

    def _decode_sensor_status(self, sum_of_codes: str):
        """
        Parses the sum of status code according to the `SENSOR_STATUSES` using
        an algorithm defined in `_get_sum_terms` method.

        Args:
            sum_of_codes
                sum of status codes, it is an integer value in the form of a
                string (e.g. "32"), as returned by the corresponding
                instrument command
        """
        codes = self._get_sum_terms(list(self.SENSOR_STATUSES.keys()),
                                    int(sum_of_codes))
        return ", ".join([self.SENSOR_STATUSES[k] for k in codes])

    @staticmethod
    def _get_sum_terms(available_terms: List[int], number: int):
        """
        Returns a list of terms which make the given number when summed up

        This method is intended to be used for cases where the given list
        of terms contains powers of 2, which corresponds to status codes
        that an instrument returns. With that said, this method is not
        guaranteed to work for an arbitrary number and an arbitrary list of
        available terms.

        Zeros are treated as a special case. If number is equal to 0,
        then [0] is returned as a list of terms. Moreover, the function
        assumes that the list of available terms contains 0 because this
        is a usually the default status code for success.

        Example:
        >>> terms = [1, 16, 32, 64, 128]
        >>> get_sum_terms(terms, 96)
        ... [64, 32]  # This is correct because 96=64+32
        """
        terms_in_number: List[int] = []

        # Sort the list of available_terms from largest to smallest
        terms_left = np.sort(available_terms)[::-1]

        # Get rid of the terms that are bigger than the number because they
        # will obviously will not make it to the returned list; and also get
        # rid of '0' as it will make the loop below infinite
        terms_left = np.array(
            [term for term in terms_left if term <= number and term != 0])

        # Handle the special case of number being 0
        if number == 0:
            terms_left = np.empty(0)
            terms_in_number = [0]

        # Keep subtracting the largest term from `number`, and always update
        # the list of available_terms so that they are always smaller than
        # the current value of `number`, until there are no more available_terms
        # to subtract
        while len(terms_left) > 0:
            term = terms_left[0]
            number -= term
            terms_in_number.append(term)
            terms_left = terms_left[terms_left <= number]

        return terms_in_number


class LakeshoreBase(VisaInstrument):
    """
    This base class has been written to be that base for the Lakeshore 336
    and 372. There are probably other lakeshore modes that can use the
    functionality provided here. If you add another lakeshore driver
    please make sure to extend this class accordingly, or create a new one.

    In order to use a variation of the `BaseSensorChannel` class for sensor
    channels, just set `CHANNEL_CLASS` to that variation of the class inside
    your `LakeshoreBase`'s subclass.

    In order to add heaters (output channels) to the driver, add `BaseOutput`
    instances (subclasses of those) in your `LakeshoreBase`'s subclass
    constructor via `add_submodule` method.
    """
    # Redefine this in the model-specific class in case you want to use a
    # different class for sensor channels
    CHANNEL_CLASS = BaseSensorChannel

    # This dict has channel name in the driver as keys, and channel "name" that
    # is used in instrument commands as values. For example, if channel called
    # "B" is referred to in instrument commands as '2', then this dictionary
    # will contain {'B': '2'} entry.
    channel_name_command: Dict[str, str] = {}

    input_channel_parameter_values_to_channel_name_on_instrument: Dict[Any, str]

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
