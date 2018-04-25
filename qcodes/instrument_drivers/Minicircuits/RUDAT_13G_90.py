import os

try:
    import clr
except ImportError:
    raise ImportError("""Module clr not found. Please obtain it by
                         running 'pip install pythonnet'
                         in a qcodes environment terminal""")

from qcodes import Instrument


class RUDAT_13G_90(Instrument):

    PATH_TO_DRIVER = r'mcl_RUDAT64'

    def __init__(self, name, driver_path=None, serial_number=None, **kwargs):
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
                f"""Load of {self.PATH_TO_DRIVER}.dll not possible. Make sure
                the dll file is not blocked by Windows. To unblock right-click
                the dll to open proporties and check the 'unblock' checkmark
                in the bottom. Check that your python installation is 64bit."""
            )

        import mcl_RUDAT64
        self._instrument = mcl_RUDAT64.USB_RUDAT()

        self._serial_number = serial_number
        if self._serial_number is None:
            ret = self._instrument.Get_Available_SN_List("")
            self._serial_number = ret[1]

        self._instrument.Connect(serial_number)
        print("connected to ", self.get_idn())

        self.add_parameter(
            "attenuation",
            set_cmd=":SETATT={}",
            get_cmd=":ATT?",
            get_parser=float
        )

    def _send_scpi(self, string):
        return self._instrument.Send_SCPI(string, "")

    def write_raw(self, cmd):
        self._send_scpi(cmd)

    def ask_raw(self, cmd):
        ret = self._send_scpi(cmd)
        return ret[1]

    def get_idn(self):
        model_name = self.ask(":MN?")
        return f"{model_name} SN: {self._serial_number}"
