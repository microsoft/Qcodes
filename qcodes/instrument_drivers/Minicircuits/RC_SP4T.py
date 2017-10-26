from qcodes import IPInstrument
from qcodes.utils import validators as vals
from qcodes.instrument.channel import InstrumentChannel, ChannelList
import math


class MC_channel(InstrumentChannel):
    def __init__(self, parent, name, channel_letter):
        """
        Args:
            parent (Instrument): The instrument the channel is a part of
            name (str): the name of the channel
            channel_letter (str): channel letter ['a', 'b'])
        """

        super().__init__(parent, name)
        self.channel_letter = channel_letter.upper()
        chanlist = ['a', 'b']
        self.channel_number = chanlist.index(channel_letter)

        self.add_parameter('switch',
                           label='switch',
                           set_cmd=self._set_switch,
                           get_cmd=self._get_switch,
                           vals=vals.Ints(0, 4)
                           )

    def _set_switch(self, switch):
        if len(self._parent.channels) > 1:
            current_switchstate = int(self.ask('SWPORT?'))
            mask = 0xF << (4 * (1 - self.channel_number))
            current_switchstate = current_switchstate & mask
        else:
            current_switchstate = 0
        # getting the current switch state of
        # the other channel if more than one channel used

        if switch == 0:
            val = 0
        else:
            val = 1 << (switch-1)
        val = val << (4 * self.channel_number)
        # only one bit in each nibble can be set
        # corrisponding to each switch state
        # setting more than one will be ignored
        val = val | current_switchstate
        self.write('SETP={}'.format(val))
        # the 'SP4T' command wont work on early devices (FW < C8)

    def _get_switch(self):
        val = int(self.ask('SWPORT?'))
        val = (val >> (4*self.channel_number)) & 0xF
        if val == 0:
            return 0
        ret = int(math.log2(val) + 1)
        return ret


class RC_SP4T(IPInstrument):
    """
    Mini-Circuits SP4T RF switch

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

        _chanlist = ['a', 'b']
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
