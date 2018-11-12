import os
from typing import Optional

# QCoDeS imports
from qcodes.instrument_drivers.Minicircuits.Base_SPDT import (
    SPDT_Base, SwitchChannelBase)

try:
    import clr
except ImportError:
    raise ImportError("""Module clr not found. Please obtain it by
                         running 'pip install pythonnet'
                         in a qcodes environment terminal""")


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
            name: the name of the instrument
            driver_path: path to the dll
            serial_number: the serial number of the device
               (printed on the sticker on the back side, without s/n)
            kwargs: kwargs to be passed to Instrument class.
    """

    CHANNEL_CLASS = SwitchChannelUSB
    PATH_TO_DRIVER = r'mcl_RF_Switch_Controller64'

    def __init__(self, name: str, driver_path: Optional[str]=None,
                 serial_number: Optional[str]=None, **kwargs):
        # we are eventually overwriting this but since it's called
        # in __getattr__ of `SPDT_Base` it's important that it's
        # always set to something to avoid infinite recursion
        self._deprecated_attributes = None
        # import .net exception so we can catch it below
        # we keep this import local so that the module can be imported
        # without a working .net install
        clr.AddReference('System.IO')
        from System.IO import FileNotFoundException
        super().__init__(name, **kwargs)
        if os.name != 'nt':
            raise ImportError("""This driver only works in Windows.""")
        try:
            if driver_path is None:
                clr.AddReference(self.PATH_TO_DRIVER)
            else:
                clr.AddReference(driver_path)

        except (ImportError, FileNotFoundException):
            raise ImportError(
                """Load of mcl_RF_Switch_Controller64.dll not possible. Make sure
                the dll file is not blocked by Windows. To unblock right-click
                the dll to open properties and check the 'unblock' checkmark
                in the bottom. Check that your python installation is 64bit."""
            )
        import mcl_RF_Switch_Controller64
        self.switch = mcl_RF_Switch_Controller64.USB_RF_SwitchBox()

        if not self.switch.Connect(serial_number):
            raise RuntimeError('Could not connect to device')
        self.address = self.switch.Get_Address()
        self.serial_number = self.switch.Read_SN('')[1]
        self.connect_message()
        self.add_channels()

    def get_idn(self):
        # the arguments in those functions is the serial number or none if
        # there is only one switch.
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
