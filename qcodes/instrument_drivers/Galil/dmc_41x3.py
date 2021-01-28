"""
This file holds the QCoDeS driver for the Galil DMC-41x3 motor controllers,
colloquially known as the "stepper motors".
"""
from typing import Any, Dict, Optional, List

from qcodes.instrument.visa import Instrument

try:
    import gclib
except ImportError as e:
    raise ImportError(
        "Cannot find gclib library. Download gclib installer from "
        "https://www.galil.com/sw/pub/all/rn/gclib.html and install Galil "
        "motion controller software for your OS. Afterwards go "
        "to https://www.galil.com/sw/pub/all/doc/gclib/html/python.html and "
        "follow instruction to be able to import gclib package in your "
        "environment.") from e


class GalilInstrument(Instrument):
    """
    Base class for Galil Motion Controller drivers
    """
    def __init__(self, name: str, address: str, **kwargs: Any) -> None:
        super().__init__(name=name, **kwargs)
        self.g = gclib.py()
        self.address = address

    def get_idn(self) -> Dict[str, Optional[str]]:
        """
        Get Galil motion controller hardware information
        """
        self.log.info('Listening for controllers requesting IP addresses...')
        ip_requests = self.g.GIpRequests()
        if len(ip_requests) != 1:
            raise RuntimeError("Multiple or No controllers connected!")

        instrument = list(ip_requests.keys())[0]
        self.log.info(instrument + " at mac" + ip_requests[instrument])

        self.log.info("Assigning " + self.address +
                      " to mac" + ip_requests[instrument])
        self.g.GAssign(self.address, ip_requests[instrument])
        self.g.GOpen(self.address + ' --direct')

        data = self.g.GInfo().split(" ")
        idparts: List[Optional[str]] = ["Galil Motion Control, Inc.",
                                        data[1], data[4], data[3][:-1]]

        return dict(zip(('vendor', 'model', 'serial', 'firmware'), idparts))

    def write_raw(self, cmd: str) -> None:
        """
        Write for Galil motion controller
        """
        self.g.GCommand(cmd+"\r")

    def ask_raw(self, cmd: str) -> str:
        """
        Asks/Reads data from Galil motion controller
        """
        return self.g.GCommand(cmd+"\r")

    def timeout(self, val: float) -> None:
        """
        Sets timeout for the instrument

        Args:
            val: time in seconds
        """
        if val < 0.001:
            raise RuntimeError("Timeout can not be less than 0.001s")

        self.g.GTimeout(val*1000)

    def close(self) -> None:
        """
        Close connection to the instrument
        """
        self.g.GClose()


class DMC4133(GalilInstrument):
    """
    Driver for Galil DMC-4133 Motor Controller
    """

    def __init__(self, name: str, address: str, **kwargs: Any) -> None:
        super().__init__(name=name, address=address, **kwargs)

        self.add_parameter("stop",
                           label="Stop Motion",
                           set_cmd="ST",
                           docstring="stop the motion of all motors")

        self.add_parameter("abort",
                           label="Abort all functions",
                           set_cmd="AB",
                           docstring="aborts motion and the program operation")
        self.connect_message()
