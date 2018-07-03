from typing import Dict, ClassVar
import logging
from collections import OrderedDict
import time
from bisect import bisect

from qcodes import VisaInstrument, InstrumentChannel, ChannelList
from qcodes.instrument.parameter import Parameter
from qcodes.utils import validators

log = logging.getLogger(__name__)

class GroupParameter(Parameter):
    def __init__(self, name, instrument, **kwargs):
        self.group = None
        super().__init__(name, instrument, **kwargs)

    def set_raw(self, value, **kwargs):  # pylint: disable=E0202
        self.group.set(self, value)

    def get_raw(self, result=None):  # pylint: disable=E0202
        if not result:
            self.group.update()
            return self.raw_value
        else:
            return result


class Group():
    def __init__(self, parameters, set_cmd=None, get_cmd=None,
                 get_parser=None, separator=',', types=None) -> None:
        self.parameters = OrderedDict((p.name, p) for p in parameters)
        self.instrument = parameters[0].root_instrument
        for p in parameters:
            p.group = self
        self.set_cmd = set_cmd
        self.get_cmd = get_cmd
        if get_parser:
            self.get_parser = get_parser
        else:
            self.get_parser = self._separator_parser(separator, types)

    def _separator_parser(self, separator, types):
        def parser(ret_str):
            keys = self.parameters.keys()
            values = ret_str.split(separator)
            return dict(zip(keys, values))
        return parser

    def set(self, set_parameter, value):
        if any((p.get_latest() is None) for p in self.parameters.values()):
            self.update()
        calling_dict = {name: p.raw_value
                        for name, p in self.parameters.items()}
        calling_dict[set_parameter.name] = value
        command_str = self.set_cmd.format(**calling_dict)
        set_parameter.root_instrument.write(command_str)

    def update(self):
        ret = self.get_parser(self.instrument.ask(self.get_cmd))
        # if any(p.get_latest() != ret[] for name, p in self.parameters if p
        # is not get_parameter):
        #     log.warn('a value has changed on the device')
        # TODO(DV): this is odd, but the only way to call the wrapper
        # accordingly
        for name, p in list(self.parameters.items()):
            p.get(result=ret[name])

VAL_MAP_TYPE = ClassVar[Dict[str, int]]
INVERSE_VAL_MAP_TYPE = ClassVar[Dict[int, str]]

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
                           val_mapping=self.MODES,
                           parameter_class=GroupParameter)
        self.add_parameter('input_channel', get_parser=int,
                           parameter_class=GroupParameter)
        self.add_parameter('powerup_enable',
                           val_mapping={True: 1, False: 0},
                           parameter_class=GroupParameter)
        self.output_group = Group([self.mode, self.input_channel,
                                  self.powerup_enable],
                                  set_cmd=f'OUTMODE {output_index}, {{mode}}, '
                                          f'{{input_channel}}, '
                                          f'{{powerup_enable}}, {{polarity}}, '
                                          f'{{filter}}, {{delay}}',
                                  get_cmd=f'OUTMODE? {output_index}')
        if self._has_pid:
            self.add_parameter('P', vals=validators.Numbers(0, 1000),
                            get_parser=float, parameter_class=GroupParameter)
            self.add_parameter('I', vals=validators.Numbers(0, 1000),
                            get_parser=float, parameter_class=GroupParameter)
            self.add_parameter('D', vals=validators.Numbers(0, 2500),
                            get_parser=float, parameter_class=GroupParameter)
            self.pid_group = Group([self.P, self.I, self.D],
                                set_cmd=f"PID {output_index}, {{P}}, {{I}}, {{D}}",
                                get_cmd=f'PID? {output_index}')

        self.add_parameter('output_range',
                           val_mapping=self.RANGES,
                           set_cmd=f'RANGE {output_index}, {{}}',
                           get_cmd=f'RANGE? {output_index}')

        self.add_parameter('setpoint',
                           vals=validators.Numbers(0, 400),
                           get_parser=float,
                           set_cmd=f'SETP {output_index}, {{}}',
                           get_cmd=f'SETP? {output_index}')

        # additional non Visa parameters
        self.add_parameter('range_limits',
                           set_cmd=None,
                           vals=validators.Sequence(validators.Numbers(0,400),
                                                    require_sorted=True,
                                                    length=7),
                           label='Temperature limits for ranges ', unit='K')

        self.add_parameter('wait_cycle_time',
                           set_cmd=None,
                           vals=validators.Numbers(0,100),
                           label='Time between two readings when waiting for'
                                 'temperature to equilibrate', unit='s')
        self.wait_cycle_time(0.1)

        self.add_parameter('wait_tolerance',
                           set_cmd=None,
                           vals=validators.Numbers(0,100),
                           label='Acceptable tolerance when waiting for '
                                 'temperature to equilibrate', unit='')
        self.wait_tolerance(0.1)

        self.add_parameter('wait_equilibration_time',
                           set_cmd=None,
                           vals=validators.Numbers(0,100),
                           label='Duration during which temperature has to be '
                                 'within tolerance.', unit='')
        self.wait_equilibration_time(0.5)


        self.add_parameter('blocking_T',
                           vals=validators.Numbers(0, 400),
                           get_parser=float,
                           set_cmd=self._set_blocking_T)

    def _set_blocking_T(self, T):
        self.set_range_from_temperature(T)
        self.setpoint(T)
        self.wait_until_set_point_reached()

    def set_range_from_temperature(self, temperature):
        """
        Sets the output range of this given heatre from from a given
        temperature.
        The output range is determine by the limits given through the parameter
        `range_limits`. The output range i is used for tempartures between
        the limits `range_limits[i-1]` and `range_limits[i]`, that is
        `range_limits` is the upper limit for using a certain heater current
        """
        if self.range_limits.get_latest() is None:
            raise RuntimeError('Error when calling set_range_from_temperature:'
                               ' You must specify the output range limits '
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

    def wait_until_set_point_reached(self, *,wait_cycle_time: float=None,
                                     wait_tolerance: float=None,
                                     wait_equilibration_time: float=None):
        wait_cycle_time =  wait_cycle_time or self.wait_cycle_time.get_latest()
        wait_tolerance = wait_tolerance or self.wait_tolerance.get_latest()
        wait_equilibration_time = (wait_equilibration_time or
                                   self.wait_equilibration_time.get_latest())
        active_channel_id = self.input_channel()
        active_channel = self.root_instrument.channels[active_channel_id]
        # assert active_channel._channel == active_channel_id

        t_setpoint = self.setpoint()
        t_reading = active_channel.temperature()
        start_time_in_tolerance_zone = None
        is_in_tolerance_zone = False
        while True:
            t_reading = active_channel.temperature()
            log.debug(f'loop iteration with t reading of {t_reading}')
            # if temperature is lower than sensor range, keep on waiting
            # TODO(DV):only do this coming from one direction
            if t_reading:
                delta = abs(t_reading-t_setpoint)/t_reading
                if delta < wait_tolerance:
                    if is_in_tolerance_zone:
                        if (time.monotonic() - start_time_in_tolerance_zone
                            > wait_equilibration_time):
                            break
                    else:
                        start_time_in_tolerance_zone = time.monotonic()
                        is_in_tolerance_zone = True
                else:
                    if is_in_tolerance_zone:
                        is_in_tolerance_zone = False
                        start_time_in_tolerance_zone = None

            log.debug(f'waiting to reach setpoint: temp at '
                      f'{t_reading}, delta:{delta}')
            time.sleep(wait_cycle_time)

class BaseSensorChannel(InstrumentChannel):

    def __init__(self, parent, name, channel):
        # args:
        #    channel: 1-4 numerical identifier of the channel
        super().__init__(parent, name)

        self._channel = channel  # Channel on the temperature controller. Can be A-D

        # Add the various channel parameters
        self.add_parameter('temperature',
                           get_cmd='KRDG? {}'.format(self._channel),
                           get_parser=float,
                           label='Temerature',
                           unit='K')

        self.add_parameter('t_limit', get_cmd=f'TLIMIT? {self._channel}',
                           set_cmd=f'TLIMIT {self._channel}, {{}}',
                           get_parser=float,
                           label='Temerature limit',
                           unit='K')

        self.add_parameter('sensor_raw',
                           get_cmd='SRDG? {}'.format(self._channel),
                           get_parser=float,
                           label='Raw_Reading',
                           unit='Ohms')  # TODO: This will vary based on sensor type

        self.add_parameter('sensor_status',
                           get_cmd='RDGST? {}'.format(self._channel),
                           val_mapping={'OK': 0,
                                        'Invalid Reading': 1,
                                        'Temp Underrange': 16,
                                        'Temp Overrange': 32,
                                        'Sensor Units Zero': 64,
                                        'Sensor Units Overrange': 128},
                           label='Sensor_Status')

        self.add_parameter('sensor_name',
                           get_cmd='INNAME? {}'.format(self._channel),
                           get_parser=str,
                           set_cmd=f'INNAME {self._channel},\"{{}}\"',
                           vals=validators.Strings(15),
                           label='Sensor_Name')




class LakeshoreBase(VisaInstrument):
    """
    This Base class has been written to be that base for the
    Lakeshore 336 and 372. There are probably other lakeshore modes that can
    use the functionality provided here. If you add another lakeshore driver
    please make sure to extend this class accordingly, or create a new one.
    """
    CHANNEL_CLASS = BaseSensorChannel

    channel_name_command: Dict[str,str] = {}

    def __init__(self, name: str, address: str,
                 terminator: str ='\r\n', **kwargs) -> None:
        super().__init__(name, address, **kwargs)
        # Allow access to channels either by referring to the channel name
        # or through a channel list.
        # i.e. instr.A.temperature() and instr.channels[0].temperature()
        # refer to the same parameter.
        self.channels = ChannelList(self, "TempSensors",
                                    self.CHANNEL_CLASS, snapshotable=False)
        for name, command in self.channel_name_command.items():
            channel = self.CHANNEL_CLASS(self, name, command)
            self.channels.append(channel)
            self.add_submodule(name, channel)
        self.channels.lock()
        self.add_submodule("channels", self.channels)

        self.connect_message()
