from qcodes import Instrument
from qcodes.utils import validators as vals
from qcodes.instrument.channel import InstrumentChannel, ChannelList
try:
    import clr
except ImportError:
    raise ImportError("""Module clr not found. Please obtain it by
                         running 'pip install -i https://pypi.anaconda.org/pythonnet/simple pythonnet' 
                         in a qcodes environment terminal""")

class SwitchChannelUSB(InstrumentChannel):
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
                           label='switch {}'.self.channel_letter,
                           set_cmd=self._set_switch,
                           get_cmd=self._get_switch,
                           vals=vals.Ints(1, 2)
                           )

    def _set_switch(self, switch): 
        self._parent.switch.Set_Switch(self.channel_letter,switch-1)

    def _get_switch(self):
        status = self._parent.switch.GetSwitchesStatus(self._parent.address)[1]
        return int("{0:04b}".format(status)[-1-self.channel_number])+1


class USB_SPDT(Instrument):
    """
    Mini-Circuits SPDT RF switch

    Args:
            name (str): the name of the instrument
            address (int, optional):
            kwargs (dict): kwargs to be passed to Instrument class.
    """
    def __init__(self, name, address=None, **kwargs):
        super().__init__(name, **kwargs)

        try:
            clr.AddReference('qcodes//instrument_drivers//Minicircuits//mcl_RF_Switch_Controller64')
        except ImportError:
            raise ImportError("""Load of mcl_RF_Switch_Controller64.dll not possible. Make sure
                                the dll file is not blocked by Windows. To unblock right-click 
                                the dll to open proporties and check the 'unblock' checkmark 
                                in the bottom. Check that your python installation is 64bit.""")
        import mcl_RF_Switch_Controller64
        self.switch = mcl_RF_Switch_Controller64.USB_RF_SwitchBox()

        if address == None:
            self.switch.Connect()
            self.address = self.switch.Get_Address()
        else:
            self.switch.ConnectByAddress(address)
            self.address = address
        self.connect_message()

        channels = ChannelList(self, "Channels", MC_channel,
                               snapshotable=False)

        _chanlist = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        _max_channel_number = int(self.IDN()['model'][4])
        _chanlist = _chanlist[0:_max_channel_number-1]

        for c in _chanlist:
            channel = SwitchChannelUSB(self, 'channel_{}'.format(c), c)
            channels.append(channel)
            self.add_submodule('channel_{}'.format(c), channel)
        channels.lock()
        self.add_submodule('channels', channels)

    def get_idn(self):
        fw = self.switch.GetFirmware()
        MN = self.switch.Read_ModelName('')[1]
        SN = self.switch.Read_SN('')[1]

        id_dict = {'firmware': fw,
                   'model': MN[3:],
                   'serial': SN[3:],
                   'vendor': 'Mini-Circuits'}
        return id_dict
