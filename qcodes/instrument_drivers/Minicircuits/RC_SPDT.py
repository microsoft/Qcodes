from qcodes import IPInstrument
from qcodes.utils import validators as vals
from qcodes.instrument.channel import InstrumentChannel, ChannelList


class MC_channel(InstrumentChannel):
    def __init__(self, parent, name, channel_letter):
        """
        Args:
            parent (Instrument): The instrument the channel is a part of
            name (str): the name of the channel
            channel_letter (str): channel letter ['a', 'b', 'c' or 'd'])
        """

        super().__init__(parent, name)
        self.channel_letter = channel_letter.upper()
        _chanlist = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        self.channel_number = _chanlist.index(channel_letter)

        self.add_parameter('switch',
                           label='switch',
                           set_cmd=self._set_switch,
                           get_cmd=self._get_switch,
                           vals=vals.Ints(1, 2)
                           )

    def _set_switch(self, switch):
        self.write('SET{}={}'.format(self.channel_letter, switch-1))

    def _get_switch(self):
        val = int(self.ask('SWPORT?'))
        # select out bit in return number
        # corisponding to channel switch configuration
        # LSB corrisponds to Chan A etc
        ret = (val >> self.channel_number) & 1
        return ret+1


class RC_SPDT(IPInstrument):
    """
    Mini-Circuits SPDT RF switch

    Args:
            name (str): the name of the instrument
            address (str): ip address ie "10.0.0.1"
            port (int): port to connect to default Telnet:23
    """
    def __init__(self, name, address, port=23):
        super().__init__(name, address, port)
        self.flush_connection()

        channels = ChannelList(self, "Channels", MC_channel,
                               snapshotable=False)

        _chanlist = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        _max_channel_number = int(self.IDN()['model'][3])
        _chanlist = _chanlist[0:_max_channel_number]

        for c in _chanlist:
            channel = MC_channel(self, 'channel_{}'.format(c), c)
            channels.append(channel)
            self.add_submodule('channel_{}'.format(c), channel)
        channels.lock()
        self.add_submodule('channels', channels)

        self.connect_message()

    def ask(self, cmd):
        ret = self.ask_raw(cmd)
        ret = ret.strip()
        return ret

    def get_idn(self):

        fw = self.ask('FIRMWARE?')
        MN = self.ask('MN?')
        SN = self.ask('SN?')

        id_dict = {'firmware': fw,
                   'model': MN[3:],
                   'serial': SN[3:],
                   'vendor': 'Mini-Circuits'}
        return id_dict
