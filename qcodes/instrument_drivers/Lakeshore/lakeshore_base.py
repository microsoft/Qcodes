from typing import Dict, Tuple

from qcodes import VisaInstrument, InstrumentChannel, ChannelList
from qcodes.utils import validators


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

        self.add_parameter('setpoint',
                           get_cmd='setp? {}'.format(self._channel),
                           set_cmd='setp {},{{}}'.format(self._channel),
                           get_parser=float,
                           label = 'Temperature setpoint',
                           unit='K')

        self.add_parameter('range_id',
                           get_cmd='range? {}'.format(self._channel),
                           set_cmd='range {},{{}}'.format(self._channel),
                           get_parser=validators.Enum(1,2,3),
                           label = 'Range ID',
                           unit='K')

        self.add_parameter('sensor_name', get_cmd='INNAME? {}'.format(self._channel),
                           get_parser=str, set_cmd='INNAME {},\"{{}}\"'.format(self._channel), vals=validators.Strings(15),
                           label='Sensor_Name')

    def set_range_from_temperature(self, temperature):
        if temperature < self._parent.t_limit(1):
            range_id = 1
        elif temperature < self._parent.t_limit(2):
            range_id = 2
        else:
            range_id = 3
        self.range_id(range_id)

    def set_setpoint_and_range(self, temperature):
        self.set_range_from_temperature(temperature)
        self.setpoint(temperature)


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

    def set_temperature_limits(self, T: Tuple[float, float]):
        self.t_limit = T

    def get_temperature_limits(self):
        return self.t_limit

    def warmup(self):
        for channel in self.channels:
            channel.temperature(300)

    def cooldown(self):
        for channel in self.channels:
            channel.temperature(0)
