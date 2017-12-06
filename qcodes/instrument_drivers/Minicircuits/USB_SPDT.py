from qcodes.instrument.visa import Instrument
try:
    import clr
except ImportError:
    raise ImportError("""Module clr not found. Please obtain it by
                         running 'pip install -i https://pypi.anaconda.org/pythonnet/simple pythonnet' 
                         in a qcodes environment terminal""")

class USB_4SPDT(Instrument):
    """
    This is a qcodes driver USB 4DSPT A18. This driver requires
    Pythonnet module to be installed in addition usual to the Qcodes environment.

    Args:
      name (str): What this instrument is called locally.
      address (int): Not needed if only one USB switch is connected to PC.
      kwargs (dict): kwargs to be passed to VisaInstrument class.
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


        ports = ['A','B','C','D']
        for port in ports:
            self.add_parameter('Port{}'.format(port),
                               label='Port {}'.format(port),
                               unit='',
                               get_cmd=getattr(self,'get_Port{}'.format(port)),
                               set_cmd=getattr(self,'set_Port{}'.format(port)),
                               get_parser=int)

    def get_PortA(self):
        return int("{0:04b}".format(self.switch.GetSwitchesStatus(self.address)[1])[-1])+1

    def get_PortB(self):
        return int("{0:04b}".format(self.switch.GetSwitchesStatus(self.address)[1])[-2])+1

    def get_PortC(self):
        return int("{0:04b}".format(self.switch.GetSwitchesStatus(self.address)[1])[-3])+1

    def get_PortD(self):
        return int("{0:04b}".format(self.switch.GetSwitchesStatus(self.address)[1])[-4])+1

    def set_PortA(self,val):
        if val in [1,2]:
            self.switch.Set_Switch("A",val-1)
        else:
            raise ValueError('Invalid input. Switch port can only be set in positions 1 or 2.')

    def set_PortB(self,val):
        if val in [1,2]:
            self.switch.Set_Switch("B",val-1)
        else:
            raise ValueError('Invalid input. Switch port can only be set in positions 1 or 2.')

    def set_PortC(self,val):
        if val in [1,2]:
            self.switch.Set_Switch("C",val-1)
        else:
            raise ValueError('Invalid input. Switch port can only be set in positions 1 or 2.')

    def set_PortD(self,val):
        if val in [1,2]:
            self.switch.Set_Switch("D",val-1)
        else:
            raise ValueError('Invalid input. Switch port can only be set in positions 1 or 2.')