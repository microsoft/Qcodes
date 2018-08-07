from qcodes.instrument.visa import VisaInstrument
from functools import partial
import logging

log = logging.getLogger(__name__)


class DynaCool(VisaInstrument):
    """
    Class to represent the DynaCoolPPMS

    Note that this driver assumes the server.py (from the 'private' folder)
    to be running on the DynaCool dedicated control PC.

    Args:
        name: The name used internaly by QCoDeS for this driver
        address: The VISA ressource name.
          E.g. 'TCPIP0::127.0.0.1::5000::SOCKET' with the appropriate IP
          address instead of 127.0.0.1
    """

    def __init__(self, name: str, address: str, **kwargs) -> None:
        super().__init__(name=name, address=address, terminator='\r\n',
                         **kwargs)

        self.add_parameter('temperature',
                           label='Temperature',
                           unit='K',
                           get_parser=partial(self._pick_one, 1, float),
                           get_cmd='TEMP?')

    def _pick_one(self, which_one: int, parser: type, resp: str) -> str:
        """
        Since most of the API calls return several values in a comma-seperated
        string, here's a convenience function to pick out the substring of
        interest
        """
        return parser(resp.split(', ')[which_one])