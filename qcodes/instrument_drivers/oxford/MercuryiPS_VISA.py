from functools import partial
from typing import Dict
import logging

from qcodes.instrument.parameter import MultiParameter
from qcodes.instrument.base import Instrument
from qcodes.instrument.channel import InstrumentChannel
from qcodes.instrument.visa import VisaInstrument

log = logging.getLogger(__name__)


def _signal_parser(response: str) -> float:
    """
    Parse a response string into a value.
    """

    # there might be a scale before the unit. We only want to deal in SI
    # units, so we translate the scale
    scale_to_factor = {'n': 1e-9, 'u': 1e-6, 'm': 1e-3,
                       '': 1, 'k': 1e3, 'M': 1e6}

    numchars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '-']

    response = response.replace(':', '')
    digits = ''.join([d for d in response if d in numchars])
    scale_and_unit = response[len(digits):]
    scale = scale_and_unit[:-1]

    return float(digits)*scale_to_factor[scale]


class MercurySlavePS(InstrumentChannel):
    """
    Class to hold a slave power supply for the MercuryiPS
    """

    def __init__(self, parent: Instrument, name: str, UID: str) -> None:
        """
        Args:
            parent: The Instrument instance of the MercuryiPS
            name: The 'colloquial' name of the PS
            UID: The UID as used internally by the MercuryiPS, e.g.
                'GRPX'
        """

        if ':' in UID:
            raise ValueError('Invalid UID. Must be axis group name or device '
                             'name, e.g. "GRPX" or "PSU.M1"')

        super().__init__(parent, name)
        self.uid = UID

        self.add_parameter('voltage',
                           label='Output voltage',
                           get_cmd=partial(self._param_getter, 'SIG:VOLT'),
                           unit='V',
                           get_parser=_signal_parser)

    def _param_getter(self, get_cmd: str) -> str:
        """
        General getter function for parameters

        Args:
            get_cmd: raw string for the command, e.g. 'SIG:VOLT'

        Returns:
            The response. Cf. MercuryiPS.ask for how much is returned
        """

        dressed_cmd = '{}:{}:{}:{}:{}'.format('READ', 'DEV', self.uid, 'PSU',
                                              get_cmd)
        resp = self._parent.ask(dressed_cmd)

        return resp


class MercuryiPS(VisaInstrument):
    """
    Driver class for the QCoDeS Oxford Instruments MercuryiPS magnet power
    supply
    """

    def __init__(self, name: str, address: str, **kwargs) -> None:
        """
        Args:
            name: The name to give this instrument internally in QCoDeS
            address: The VISA resource of the instrument. Note that a
                socket connection to port 7020 must be made
        """

        # ensure that a socket is used
        if not address.endswith('SOCKET'):
             ValueError('Incorrect VISA resource name. Must be of type '
                             'TCPIP0::XXX.XXX.XXX.XXX::7020::SOCKET.')

        super().__init__(name, address, terminator='\n', **kwargs)

        # to ensure a correct snapshot, we must wrap the get function
        self.IDN.get = self.IDN._wrap_get(self._idn_getter)

        # TODO: Query instrument to ensure which PSUs we have
        for grp in ['GRPX', 'GRPY', 'GRPZ']:
            psu_name = grp.replace('GRP', '')
            psu = MercurySlavePS(self, psu_name, grp)
            self.add_submodule(psu_name, psu)

        self.connect_message()

    def _idn_getter(self) -> Dict[str, str]:
        """
        Parse the raw non-SCPI compliant IDN string into an IDN dict

        Returns:
            The normal IDN dict
        """
        raw_idn_string = self.ask('*IDN?')
        resps = raw_idn_string.split(':')

        idn_dict = {'model': resps[2], 'vendor': resps[1],
                    'serial': resps[3], 'firmware': resps[4]}

        # idn_string = ','.join([resps[2], resps[1], resps[3], resps[4]])

        return idn_dict

    def ask(self, cmd: str) -> str:
        """
        Since Oxford Instruments implement their own version of a SCPI-like
        language, we implement our own reader.

        Args:
            cmd: the command to send to the instrument
        """

        resp = self.visa_handle.ask(cmd)

        if 'INVALID' in resp:
            log.error('Invalid command. Got response: {}'.format(resp))
            base_resp = resp
        else:
            # For "normal" commands only (e.g. '*IDN?' is excepted):
            # the response of a valid command echoes back said command,
            # thus we remove that part
            base_cmd = cmd.replace('READ:', '')
            base_resp = resp.replace('STAT:{}'.format(base_cmd), '')

            return base_resp
