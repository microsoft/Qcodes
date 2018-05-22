from typing import Dict, Tuple

from qcodes import VisaInstrument, InstrumentChannel, ChannelList
from qcodes.instrument.parameter import Parameter
from qcodes.utils import validators


class GroupParameter(Parameter):
    def __init__(self, name, instrument, **kwargs):
        self.group = None
        super().__init__(name, instrument, **kwargs)

    def set_raw(self, value, **kwargs):
        self.group.set(self, value)

    def get_raw(self, result=None):
        if not result:
            self.group.update()
            return self.raw_value
        else:
            return result


class Group():
    def __init__(self, parameters, set_cmd=None, get_cmd=None,
                 get_parser=None, separator=',', types=None):
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
            print(f'retr string is {ret_str}')
            keys = self.parameters.keys()
            values = ret_str.split(separator)
            return dict(zip(keys, values))
        return parser
    
    def set(self, set_parameter, value):
        if any((p.get_latest() is None) for p in self.parameters.values()):
            self.update()
        calling_dict = {name: p.raw_value for name, p in self.parameters.items()}
        calling_dict[set_parameter.name] = value
        command_str = self.set_cmd.format(**calling_dict)
        set_parameter.root_instrument.write(command_str)

    def update(self):
        ret = self.get_parser(self.instrument.ask(self.get_cmd))
        # if any(p.get_latest() != ret[] for name, p in self.parameters if p is not get_parameter):
        #     log.warn('a value has changed on the device')
        # TODO(DV): this is odd, but the only way to call the wrapper accordingly
        for name, p in list(self.parameters.items()):
            p.get(result=ret[name])
        

class BaseOutput(InstrumentChannel):

    MODES = {}
    RANGES = {}

    def __init__(self, parent, output_name, output_index):
        super().__init__(parent, output_name)
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
                                  set_cmd=f"OUTMODE {output_index}, {{mode}}, {{input_channel}}, {{powerup_enable}}, {{polarity}}, {{filter}}, {{delay}}",
                                  get_cmd=f'OUTMODE? {output_index}')

        self.add_parameter('P', vals=vals.Numbers(0, 1000),
                           get_parser=float, parameter_class=GroupParameter)
        self.add_parameter('I', vals=vals.Numbers(0, 1000),
                           get_parser=float, parameter_class=GroupParameter)
        self.add_parameter('D', vals=vals.Numbers(0, 2500),
                           get_parser=float, parameter_class=GroupParameter)
        self.pid_group = Group([self.P, self.I, self.D],
                               set_cmd=f"PID {output_index}, {{P}}, {{I}}, {{D}}",
                               get_cmd=f'PID? {output_index}')
        
        self.add_parameter('range',
                           val_mapping=self.RANGES,
                           set_cmd=f'RANGE {output_index}, {{}}',
                           get_cmd=f'RANGE? {output_index}')

        self.add_parameter('setpoint', 
                           vals=vals.Numbers(0, 400),
                           get_parser=float,
                           set_cmd=f'SETP {output_index}, {{}}',
                           get_cmd=f'SETP? {output_index}')


class BaseSensorChannel(InstrumentChannel):

    def __init__(self, parent, name, channel):
        # args:
        #    channel: 1-4 numerical identifier of the channel
        super().__init__(parent, name)

        self._channel = channel  # Channel on the temperature controller. Can be A-D

        # Add the various channel parameters
        self.add_parameter('temperature', get_cmd='KRDG? {}'.format(self._channel),
                           get_parser=float,
                           label='Temerature',
                           unit='K')
        self.add_parameter('sensor_raw', get_cmd='SRDG? {}'.format(self._channel),
                           get_parser=float,
                           label='Raw_Reading',
                           unit='Ohms')  # TODO: This will vary based on sensor type

        self.add_parameter('sensor_status', get_cmd='RDGST? {}'.format(self._channel),
                           val_mapping={'OK': 0, 'Invalid Reading': 1, 'Temp Underrange': 16, 'Temp Overrange': 32,
                           'Sensor Units Zero': 64, 'Sensor Units Overrange': 128}, label='Sensor_Status')

        self.add_parameter('sensor_name', get_cmd='INNAME? {}'.format(self._channel),
                           get_parser=str, set_cmd='INNAME {},\"{{}}\"'.format(self._channel), vals=validators.Strings(15),
                           label='Sensor_Name')



class LakeshoreBase(VisaInstrument):
    """
    This Base class has been written to be that base for the Lakeshore 336 and 372. There are probably other lakeshore modes that can use the functionality provided here. If you add another lakeshore driver please make sure to extend this class accordingly, or create a new one.
    """
    CHANNEL_CLASS = BaseSensorChannel

    channel_name_command: Dict[str,str] = {'A': 'A',
                                           'B': 'B',
                                           'C': 'C',
                                           'D': 'D'}

    def __init__(self, name: str, address: str,
                 terminator: str ='\r\n', **kwargs):
        super().__init__(name, address, **kwargs)
        self.add_parameter('temperature_limits',
                           set_cmd=self.set_temperature_limits,
                           get_cmd=self.get_temperature_limits,
                           label='Temperature limits for ranges ',
                           unit='K')

        # plug some senisble values in here
        self.t_limit: Tuple[float, float] = (10.0, 20.0)

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

    # def set_temperature_limits(self, T: Tuple[float, float]):
    #     self.t_limit = T

    # def get_temperature_limits(self):
    #     return self.t_limit

    # def warmup(self):
    #     for channel in self.channels:
    #         channel.temperature(300)

    # def cooldown(self):
    #     for channel in self.channels:
    #         channel.temperature(0)
