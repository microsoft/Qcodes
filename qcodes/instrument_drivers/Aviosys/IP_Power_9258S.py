from qcodes import Instrument, InstrumentChannel, ChannelList
from qcodes.utils.validators import Enum
from qcodes.utils.helpers import create_on_off_val_mapping
import urllib


class PowerChannel(InstrumentChannel):
    """
    Channel class for a socket on the Aviosys IP Power 9258S.
    Args:
        parent (Instrument): Parent instrument.
        name (str): Channel name.
        channel (str): Alphabetic channel id.
    Attributes:
        channel (str): Alphabetic channel id.
        channel_id (int): Numeric channel id.
    """

    _channel_values = Enum('A', 'B', 'C', 'D')
    _channel_ids = {'A': 1, 'B': 2, 'C': 3, 'D': 4}

    def __init__(self, parent, name, channel):

        super().__init__(parent, name)

        # validate the channel id
        self._channel_values.validate(channel)
        self.channel = channel
        self.channel_id = self._channel_ids[channel]

        # add parameters
        self.add_parameter('power',
                           get_cmd=self.get_power,
                           set_cmd=self.set_power,
                           get_parser=int,
                           val_mapping=create_on_off_val_mapping(on_val=1, off_val=0),
                           label='power {}'.format(self.channel))

    # get methods
    def get_power(self):
        request = urllib.request.Request(self.parent.address+'/set.cmd?cmd=getpower')
        response = urllib.request.urlopen(request)
        request_read = response.read()
        return request_read.decode("utf-8")[4+int(self.channel_id)*6]

    # set methods
    def set_power(self, power):
        request = urllib.request.Request(self.parent.address+'/set.cmd?cmd=setpower+p6%s=%s' % (self.channel_id, power))
        urllib.request.urlopen(request)


class Aviosys_IP_Power_9258S(Instrument):
    """
    Instrument driver for the Aviosys IP Power 9258 remote socket controller.
    Args:
        name (str): Instrument name.
        address (str): http address.
        login_name (str): http login name.
        login_password (str) http login password.
    Attributes:
        address (str): http address.
    """

    def __init__(self, name, address, login_name, login_password, **kwargs):

        super().__init__(name, **kwargs)

        # save access settings
        self.address = address

        # set up http connection
        password_manager = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        password_manager.add_password(None, self.address, login_name, login_password)
        handler = urllib.request.HTTPBasicAuthHandler(password_manager)
        opener = urllib.request.build_opener(handler)
        urllib.request.install_opener(opener)

        # add channels
        channels = ChannelList(self, "PowerChannels", PowerChannel, snapshotable=False)
        for channel_id in ('A', 'B', 'C', 'D'):
            channel = PowerChannel(self, 'Chan{}'.format(channel_id), channel_id)
            channels.append(channel)
            self.add_submodule(channel_id, channel)
        channels.lock()
        self.add_submodule("channels", channels)

        # print connect message
        self.connect_message()

    # get functions
    def get_idn(self):
        return {'vendor': 'Aviosys', 'model': 'IP Power 9258S'}
