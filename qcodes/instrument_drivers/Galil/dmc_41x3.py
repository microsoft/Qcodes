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
        "https://www.galil.com/sw/pub/all/rn/gclib.html for your OS and "
        "install Galil motion controller software for your OS. Afterwards go "
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
        ips = {}
        self.log.info('Listening for controllers requesting IP addresses...')
        ip_requests = self.g.GIpRequests()
        for i in ip_requests.keys():
            self.log.info(i + ' at mac' + ip_requests[i])
            ips[i] = self.address

        for i in ips.keys():
            if i in ip_requests:
                self.log.info("Assigning " + ips[i] + " to " + ip_requests[i])
                self.g.GAssign(ips[i], ip_requests[i])
                self.g.GOpen(ips[i] + ' --direct')
                self.log.info(self.g.GInfo())

        available = self.g.GAddresses()
        data = available[self.address].split(" ")
        idparts: List[Optional[str]] = ["Galil Motion Control, Inc.",
                                        data[0], None, data[2]]

        return dict(zip(('vendor', 'model', 'serial', 'firmware'), idparts))

    def write_raw(self, cmd: str) -> None:
        """
        Write for Galil motion controller
        """
        self.g.GCommand(cmd)

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

        self.connect_message()
