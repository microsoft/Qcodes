from enum import Enum

from qcodes import VisaInstrument, InstrumentChannel, ChannelList
from qcodes.utils.validators import Enum as QCEnum, Strings

class ChannelDescriptor(Enum):
    A = 1
    B = 2
    C = 3
    D = 4

class SensorChannel(InstrumentChannel):
    """
    A single sensor channel of a temperature controller
    """

    # _CHANNEL_VAL = Enum("A", "B", "C", "D")
    _CHANNEL_VAL = QCEnum([c.value for c in list(ChannelDescriptor)])


    def __init__(self, parent, name, channel):
        # args:
        #    channel: 1-4 numerical identifier of the channel
        super().__init__(parent, name)

        # Validate the channel value
        self._CHANNEL_VAL.validate(channel)
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

        self.add_parameter('range',
                           get_cmd='range? {}'.format(self._channel),
                           set_cmd='range {},{{}}'.format(self._channel),
                           get_parser=float,
                           label = 'Temperature setpoint',
                           unit='K')

        self.add_parameter('sensor_name', get_cmd='INNAME? {}'.format(self._channel),
                           get_parser=str, set_cmd='INNAME {},\"{{}}\"'.format(self._channel), vals=Strings(15),
                           label='Sensor_Name')


class Model_336(VisaInstrument):
    """
    Lakeshore Model 336 Temperature Controller Driver
    Controlled via sockets
    """

    def __init__(self, name, address, **kwargs):
        if 'terminator' not in kwargs:
            kwargs['terminator'] = "\r\n"
        super().__init__(name, address, **kwargs)

        self.add_parameter('temperature_limits',
                            set_cmd=self.set_temperature_limits,
                            get_cmd=self.get_temperature_limits,
                            label='Temperature limits for ranges ',
                            unit='K')

        # plug some senisble values in here
        self.t_limit = (1, 2)

        # Allow access to channels either by referring to the channel name
        # or through a channel list.
        # i.e. Model_336.A.temperature() and Model_336.channels[0].temperature()
        # refer to the same parameter.
        self.channels = ChannelList(self, "TempSensors", SensorChannel, snapshotable=False)
        for c in list(ChannelDescriptor):
            channel = SensorChannel(self, 'Chan{}'.format(c.name), c.value)
            self.channels.append(channel)
            self.add_submodule(c.name, channel)
        channels.lock()
        self.add_submodule("channels", self.channels)

        self.connect_message()

    def set_temperature_limits(self, T):
        self.t_limit = T

    def get_temperature_limits(self):
        return self.t_limit

    def warmup(self):
        for channel in self.channels:
            channel.temperature(300)

    def cooldown(self):
        for channel in self.channels:
            channel.temperature(0)
