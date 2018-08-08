from qcodes.instrument.visa import VisaInstrument
from functools import partial
import logging
from typing import Dict, Optional

from visa import VisaIOError

import qcodes.utils.validators as vals

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

    def __init__(self, name: str,
                 address: str,
                 **kwargs) -> None:
        super().__init__(name=name, address=address, terminator='\r\n',
                         **kwargs)

        self.add_parameter('temperature',
                           label='Temperature',
                           unit='K',
                           get_parser=partial(DynaCool._pick_one, 1, float),
                           get_cmd='TEMP?')

        self.add_parameter('temperature_setpoint',
                           label='Temperature setpoint',
                           unit='K',
                           vals=vals.Numbers(1.8, 400),
                           set_cmd=partial(self._temp_setter,
                                           'temperature_setpoint'))

        self.add_parameter('temperature_rate',
                           label='Temperature settle rate',
                           unit='K/s',
                           vals=vals.Numbers(0.0002, 0.3),
                           set_parser=lambda x: x*60,  # conversion to K/min
                           set_cmd=partial(self._temp_setter,
                                           'temperature_rate'))

        self.add_parameter('temperature_settling',
                           label='Temperature settling mode',
                           val_mapping={'fast settle': 0,
                                        'no overshoot': 1},
                           set_cmd=partial(self._temp_setter,
                                           'temperature_settling'))

        self.add_parameter('chamber_temperature',
                           label='Chamber Temperature',
                           unit='K',
                           get_parser=partial(DynaCool._pick_one, 1, float),
                           get_cmd='CHAT?')

        # The error code of the latest command
        self._error_code = 0

        self._update_temperatures()

        self.connect_message()

    @property
    def error_code(self):
        return self._error_code

    @staticmethod
    def _pick_one(which_one: int, parser: type, resp: str) -> str:
        """
        Since most of the API calls return several values in a comma-seperated
        string, here's a convenience function to pick out the substring of
        interest
        """
        return parser(resp.split(', ')[which_one])

    def get_idn(self) -> Dict[str, Optional[str]]:
        response = self.ask('*IDN?')
        # just clip out the error code
        idparts = response[2:].split(', ')

        return dict(zip(('vendor', 'model', 'serial', 'firmware'), idparts))

    def _update_temperatures(self) -> None:
        """
        This function queries the last temperature setpoint (w. rate and mode)
        from the instrument.
        """
        raw_response = self.ask('GLTS?')
        sp = DynaCool._pick_one(1, float, raw_response)
        rate = DynaCool._pick_one(2, float, raw_response)
        mode = DynaCool._pick_one(3, int, raw_response)

        self.temperature_setpoint._save_val(sp)
        self.temperature_rate.raw_value = rate
        self.temperature_rate._save_val(rate/60)
        self.temperature_settling.raw_value = mode
        inv_map = {val: key for (key, val) in
                   self.temperature_settling.val_mapping.items()}
        self.temperature_settling._save_val(inv_map[mode])

    def _temp_setter(self, param: str, value: float) -> None:
        self._update_temperatures()
        params = ['temperature_setpoint', 'temperature_rate',
                  'temperature_settling']
        vals = list(self.parameters[par].raw_value for par in params)
        vals[params.index(param)] = value

        self.write(f'TEMP {vals[0]}, {vals[1]}, {vals[2]}')

    def write(self, cmd: str) -> None:
        """
        Since the error code is always returned, we must read it back
        """
        super().write(cmd)
        self._error_code = self.visa_handle.read()

    def ask(self, cmd: str) -> str:
        """
        Since the error code is always returned, we must read it back
        """
        response = super().ask(cmd)
        self._error_code = DynaCool._pick_one(0, str, response)
        return response

    def close(self) -> None:
        """
        Make sure to nicely close the server connection
        """
        try:
            log.debug('Closing server connection.')
            self.write('CLOSE')
        except VisaIOError:
            log.info('Could not close connection to server, perhaps the '
                     'server is down?')
        super().close()
