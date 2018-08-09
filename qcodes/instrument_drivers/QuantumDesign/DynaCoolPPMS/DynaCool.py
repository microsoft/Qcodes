from qcodes.instrument.visa import VisaInstrument
from functools import partial
import logging
from typing import Dict, Optional, Union

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

    temp_params = ['temperature_setpoint', 'temperature_rate',
                   'temperature_settling']

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
                                           'temperature_setpoint'),
                           get_cmd=partial(self._temp_getter,
                                           'temperature_setpoint'))

        self.add_parameter('temperature_rate',
                           label='Temperature settle rate',
                           unit='K/s',
                           vals=vals.Numbers(0.0002, 0.3),
                           set_parser=lambda x: x*60,  # conversion to K/min
                           get_parser=lambda x: x/60,  # conversion to K/s
                           set_cmd=partial(self._temp_setter,
                                           'temperature_rate'),
                           get_cmd=partial(self._temp_getter,
                                           'temperature_rate'))

        self.add_parameter('temperature_settling',
                           label='Temperature settling mode',
                           val_mapping={'fast settle': 0,
                                        'no overshoot': 1},
                           set_cmd=partial(self._temp_setter,
                                           'temperature_settling'),
                           get_cmd=partial(self._temp_getter,
                                           'temperature_settling'))

        self.add_parameter('temperature_state',
                           label='Temperature tracking state',
                           val_mapping={"tracking": 2,
                                        'stable': 1},
                           get_parser=partial(DynaCool._pick_one, 2, int),
                           get_cmd='TEMP?')

        self.add_parameter('chamber_temperature',
                           label='Chamber Temperature',
                           unit='K',
                           get_parser=partial(DynaCool._pick_one, 1, float),
                           get_cmd='CHAT?')

        # The error code of the latest command
        self._error_code = 0

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

    def _temp_getter(self, param_name: str) -> Union[int, float]:
        """
        This function queries the last temperature setpoint (w. rate and mode)
        from the instrument.
        """
        raw_response = self.ask('GLTS?')
        sp = DynaCool._pick_one(1, float, raw_response)
        rate = DynaCool._pick_one(2, float, raw_response)
        mode = DynaCool._pick_one(3, int, raw_response)

        return dict(zip(self.temp_params, [sp, rate, mode]))[param_name]

    def _temp_setter(self, param: str, value: float) -> None:
        """
        The setter function for the temperature parameters. All three are set
        with the same call to the instrument API
        """
        vals = list(self.parameters[par].raw_value for par in self.temp_params)
        vals[self.temp_params.index(param)] = value

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
