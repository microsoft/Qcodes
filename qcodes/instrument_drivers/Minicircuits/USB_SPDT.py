import logging
import os
import re
import warnings

# QCoDeS imports
from qcodes import Instrument
from qcodes.instrument.channel import ChannelList, InstrumentChannel
from qcodes.utils import validators as vals

try:
    import clr
except ImportError:
    raise ImportError("""Module clr not found. Please obtain it by
                         running 'pip install pythonnet'
                         in a qcodes environment terminal""")

log = logging.getLogger(__name__)


class SwitchChannelBase(InstrumentChannel):
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

        self.add_parameter(
            'switch',
            label='switch {}'.format(self.channel_letter),
            set_cmd=self._set_switch,
            get_cmd=self._get_switch,
            vals=vals.Ints(1, 2))

    def __call__(self, *args):
        if len(args) == 1:
            self._set_switch(args[0])
        elif len(args) == 0:
            return self._get_switch()
        else:
            raise RuntimeError(
                'Call channel with either one or zero arguments')

    def _set_switch(self, switch):
        raise NotImplementedError()

    def _get_switch(self):
        raise NotImplementedError()


class SPDT_Base(Instrument):
    @property
    def CHANNEL_CLASS(self):
        raise NotImplementedError

    def add_channels(self):
        channels = ChannelList(
            self, "Channels", self.CHANNEL_CLASS, snapshotable=False)

        self._deprecated_attributes = {}

        _chanlist = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        _max_channel_number = self.get_number_of_channels()
        _chanlist = _chanlist[0:_max_channel_number]

        for c in _chanlist:
            channel = self.CHANNEL_CLASS(self, 'channel_{}'.format(c), c)
            channels.append(channel)
            attribute_name = 'channel_{}'.format(c)
            self.add_submodule(attribute_name, channel)
            self.add_submodule(c, channel)
            self._deprecated_attributes[attribute_name] = c
        channels.lock()
        self.add_submodule('channels', channels)

    def all(self, switch_to):
        for c in self.channels:
            c.switch(switch_to)

    def __getattr__(self, key):
        if key in self._deprecated_attributes:
            warnings.warn("""Using '{}' is deprecated and will be removed in
                future releases. Use '{}' instead""".format(
                key, self._deprecated_attributes[key]), DeprecationWarning)
        return super().__getattr__(key)

    def get_number_of_channels(self):
        model = self.get_idn()['model']
        model_parts = model.split('-')
        if len(model_parts) < 2:
            raise RuntimeError('The driver could not determine the number of channels of the model \'{}\', it might not be supported')
        if model_parts[0] not in ('RC', 'USB'):
            log.warning('The model with the name \'{}\' might not be supported by the driver'.format(model))
        channels = re.match('^[0-9]*', model_parts[1])[0]
        if not channels:
            raise RuntimeError('The driver could not determine the number of channels of the model \'{}\', it might not be supported')
        return int(channels)


class SwitchChannelUSB(SwitchChannelBase):

    def _set_switch(self, switch):
        self._parent.switch.Set_Switch(self.channel_letter, switch - 1)

    def _get_switch(self):
        status = self._parent.switch.GetSwitchesStatus(self._parent.address)[1]
        return int("{0:04b}".format(status)[-1 - self.channel_number]) + 1


class USB_SPDT(SPDT_Base):
    """
    Mini-Circuits SPDT RF switch

    Args:
            name (str): the name of the instrument
            address (int, optional):
            kwargs (dict): kwargs to be passed to Instrument class.
    """

    CHANNEL_CLASS = SwitchChannelUSB
    PATH_TO_DRIVER = r'mcl_RF_Switch_Controller64'

    def __init__(self, name, driver_path=None, address=None, **kwargs):
        super().__init__(name, **kwargs)
        if os.name != 'nt':
            raise ImportError("""This driver only works in Windows.""")
        try:
            if driver_path is None:
                clr.AddReference(self.PATH_TO_DRIVER)
            else:
                clr.AddReference(driver_path)

        except ImportError:
            raise ImportError(
                """Load of mcl_RF_Switch_Controller64.dll not possible. Make sure
                the dll file is not blocked by Windows. To unblock right-click
                the dll to open proporties and check the 'unblock' checkmark
                in the bottom. Check that your python installation is 64bit."""
            )
        import mcl_RF_Switch_Controller64
        self.switch = mcl_RF_Switch_Controller64.USB_RF_SwitchBox()

        if address is None:
            self.switch.Connect()
            self.address = self.switch.Get_Address()
        else:
            self.switch.ConnectByAddress(address)
            self.address = address
        self.connect_message()
        self.add_channels()

    def get_idn(self):
        fw = self.switch.GetFirmware()
        MN = self.switch.Read_ModelName('')[1]
        SN = self.switch.Read_SN('')[1]

        id_dict = {
            'firmware': fw,
            'model': MN,
            'serial': SN,
            'vendor': 'Mini-Circuits'
        }
        return id_dict
