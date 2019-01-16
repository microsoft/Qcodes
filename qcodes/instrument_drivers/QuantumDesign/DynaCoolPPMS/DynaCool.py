from qcodes.instrument.visa import VisaInstrument
from functools import partial
import logging
from typing import Dict, Optional, Union, cast, Any, List
import warnings

from visa import VisaIOError

import qcodes.utils.validators as vals

class DynaCool(VisaInstrument):
    """
    Class to represent the DynaCoolPPMS

    Note that this driver assumes the server.py (from the 'private' folder)
    to be running on the DynaCool dedicated control PC.

    Args:
        name: The name used internaly by QCoDeS for this driver
        address: The VISA resource name.
          E.g. 'TCPIP0::127.0.0.1::5000::SOCKET' with the appropriate IP
          address instead of 127.0.0.1. Note that the port number is
          hard-coded into the server.
    """

    temp_params = ['temperature_setpoint', 'temperature_rate',
                   'temperature_settling']
    field_params = ['field_setpoint', 'field_rate', 'field_approach']

    _errors = {-2: lambda: warnings.warn('Unknown command'),
               1: lambda: None,
               0: lambda: None}

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

        # Note: from the Lyngby Materials Lab, we have been told that the
        # manual is wrong about the minimal temperature. The manual says
        # 1.8 K, but it is in fact 1.6 K
        self.add_parameter('temperature_setpoint',
                           label='Temperature setpoint',
                           unit='K',
                           vals=vals.Numbers(1.6, 400),
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
                                        'stable': 1,
                                        'near': 5,
                                        'chasing': 6,
                                        'pot operation': 7,
                                        'standby': 10,
                                        'diagnostic': 13,
                                        'impedance control error': 14,
                                        'failure': 15},
                           get_parser=partial(DynaCool._pick_one, 2, int),
                           get_cmd='TEMP?')

        self.add_parameter('field',
                           label='Field strength',
                           unit='A/m',
                           get_cmd=self._present_field_getter)

        self.add_parameter('field_setpoint',
                           label='Field setpoint',
                           unit='A/m',
                           get_parser=lambda x: x*1e-4,  # Oe to T
                           set_parser=lambda x: x*1e4,  # T to Oe
                           set_cmd=partial(self._field_setter,
                                           'field_setpoint'),
                           get_cmd=partial(self._field_getter,
                                           'field_setpoint'),
                           vals=vals.Numbers(-1755, 1755))

        self.add_parameter('field_rate',
                           label='Field rate',
                           unit='A/m/s',
                           get_parser=lambda x: x*1e-4,  # Oe to T
                           set_parser=lambda x: x*1e4,  # T to Oe
                           set_cmd=partial(self._field_setter,
                                           'field_rate'),
                           get_cmd=partial(self._field_getter,
                                           'field_rate'),
                           vals=vals.Numbers(-125.6, 125.6))

        self.add_parameter('field_approach',
                           label='Field ramp approach',
                           val_mapping={'linear': 0,
                                        'no overshoot': 1,
                                        'oscillate': 2},
                           set_cmd=partial(self._field_setter,
                                           'field_approach'),
                           get_cmd=partial(self._field_getter,
                                           'field_approach'))

        self.add_parameter('magnet_state',
                           label='Magnet state',
                           val_mapping={'unknown': 0,
                                        'stable': 1,
                                        'switch warming': 2,
                                        'switch cool': 3,
                                        'holding': 4,
                                        'iterate': 5,
                                        'ramping': 6,
                                        'ramping ': 7,  # map must be bijection
                                        'resetting': 8,
                                        'current error': 9,
                                        'switch error': 10,
                                        'quenching': 11,
                                        'charging error': 12,
                                        'power supply error': 14,
                                        'failure': 15},
                           get_parser=partial(DynaCool._pick_one, 2, int),
                           get_cmd='FELD?')

        self.add_parameter('chamber_temperature',
                           label='Chamber Temperature',
                           unit='K',
                           get_parser=partial(DynaCool._pick_one, 1, float),
                           get_cmd='CHAT?')

        self.add_parameter('chamber_state',
                           label='Chamber vacuum state',
                           val_mapping={'purged and sealed': 1,
                                        'vented and sealed': 2,
                                        'sealed': 3,
                                        'performing purge/seal': 4,
                                        'performing vent/seal': 5,
                                        'pre-high vacuum': 6,
                                        'high vacuum': 7,
                                        'pumping continuously': 8,
                                        'flooding continuously': 9},
                           get_parser=partial(DynaCool._pick_one, 1, int),
                           get_cmd='CHAM?')

        # The error code of the latest command
        self._error_code = 0

        # we must know all parameter values because of interlinked parameters
        self.snapshot(update=True)

        self.connect_message()

    @property
    def error_code(self):
        return self._error_code

    @staticmethod
    def _pick_one(which_one: int, parser: type, resp: str) -> Any:
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

    def _present_field_getter(self) -> float:
        resp = self.ask('FELD?')
        number_in_oersted = cast(float, DynaCool._pick_one(1, float, resp))
        number_in_SI = number_in_oersted/79.57747
        return number_in_SI

    def _field_getter(self, param_name: str) -> Union[int, float]:
        """
        The combined get function for the three field parameters,
        field_setpoint, field_rate, and field_approach
        """
        raw_response = self.ask('GLFS?')
        sp = DynaCool._pick_one(1, float, raw_response)
        rate = DynaCool._pick_one(2, float, raw_response)
        approach = DynaCool._pick_one(3, int, raw_response)

        return dict(zip(self.field_params, [sp, rate, approach]))[param_name]

    def _field_setter(self, param: str, value: float) -> None:
        """
        The combined set function for the three field parameters,
        field_setpoint, field_rate, and field_approach
        """
        temp_values = list(self.parameters[p].raw_value
                           for p in self.field_params)
        values = cast(List[Union[int, float]], temp_values)
        values[self.field_params.index(param)] = value

        self.write(f'FELD {values[0]}, {values[1]}, {values[2]}, 0')

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
        temp_values = list(self.parameters[par].raw_value
                           for par in self.temp_params)
        values = cast(List[Union[int, float]], temp_values)
        values[self.temp_params.index(param)] = value

        self.write(f'TEMP {values[0]}, {values[1]}, {values[2]}')

    def write(self, cmd: str) -> None:
        """
        Since the error code is always returned, we must read it back
        """
        super().write(cmd)
        self._error_code = int(self.visa_handle.read())
        self._errors[self._error_code]()
        self.visa_log.debug(f'Error code: {self._error_code}')

    def ask(self, cmd: str) -> str:
        """
        Since the error code is always returned, we must read it back
        """
        response = super().ask(cmd)
        self._error_code = DynaCool._pick_one(0, int, response)
        self._errors[self._error_code]()
        return response

    def close(self) -> None:
        """
        Make sure to nicely close the server connection
        """
        try:
            self.log.debug('Closing server connection.')
            self.write('CLOSE')
        except VisaIOError as e:
            self.log.info('Could not close connection to server, perhaps the '
                          'server is down?')
            self.log.info(f'Got the following error from PyVISA: '
                          f'{e.abbreviation}: {e.description}')
        super().close()
