import warnings
from functools import partial
from time import sleep
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Literal,
    Optional,
    Union,
    cast,
)

import numpy as np
from pyvisa import VisaIOError

import qcodes.validators as vals
from qcodes.instrument import VisaInstrument, VisaInstrumentKWArgs

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from qcodes.parameters import Parameter


class DynaCool(VisaInstrument):
    """
    Class to represent the DynaCoolPPMS

    Note that this driver assumes the server
    to be running on the DynaCool dedicated control PC.
    The server can be launched using `qcodes-dynacool-server.exe`
    or by executing server.py (from the 'private' folder)

    Args:
        name: The name used internally by QCoDeS for this driver
        address: The VISA resource name.
          E.g. 'TCPIP0::127.0.0.1::5000::SOCKET' with the appropriate IP
          address instead of 127.0.0.1. Note that the port number is
          hard-coded into the server.
    """

    # the ramp time resolution is in (s) and is used in the
    # _do_blocking_ramp method
    _ramp_time_resolution = 0.1

    temp_params = ("temperature_setpoint", "temperature_rate", "temperature_settling")
    field_params = ("field_target", "field_rate", "field_approach")

    _errors: ClassVar[dict[int, Callable[[], None]]] = {
        -2: lambda: warnings.warn("Unknown command"),
        1: lambda: None,
        0: lambda: None,
    }

    default_terminator = "\r\n"

    def __init__(
        self, name: str, address: str, **kwargs: "Unpack[VisaInstrumentKWArgs]"
    ) -> None:
        super().__init__(name=name, address=address, **kwargs)

        self.temperature: Parameter = self.add_parameter(
            "temperature",
            label="Temperature",
            unit="K",
            get_parser=partial(DynaCool._pick_one, 1, float),
            get_cmd="TEMP?",
        )
        """Parameter temperature"""

        # Note: from the Lyngby Materials Lab, we have been told that the
        # manual is wrong about the minimal temperature. The manual says
        # 1.8 K, but it is in fact 1.6 K
        self.temperature_setpoint: Parameter = self.add_parameter(
            "temperature_setpoint",
            label="Temperature setpoint",
            unit="K",
            vals=vals.Numbers(1.6, 400),
            set_cmd=partial(self._temp_setter, "temperature_setpoint"),
            get_cmd=partial(self._temp_getter, "temperature_setpoint"),
        )
        """Parameter temperature_setpoint"""

        self.temperature_rate: Parameter = self.add_parameter(
            "temperature_rate",
            label="Temperature settle rate",
            unit="K/s",
            vals=vals.Numbers(0.0002, 0.3),
            set_parser=lambda x: x * 60,  # conversion to K/min
            get_parser=lambda x: x / 60,  # conversion to K/s
            set_cmd=partial(self._temp_setter, "temperature_rate"),
            get_cmd=partial(self._temp_getter, "temperature_rate"),
        )
        """Parameter temperature_rate"""

        self.temperature_settling: Parameter = self.add_parameter(
            "temperature_settling",
            label="Temperature settling mode",
            val_mapping={"fast settle": 0, "no overshoot": 1},
            set_cmd=partial(self._temp_setter, "temperature_settling"),
            get_cmd=partial(self._temp_getter, "temperature_settling"),
        )
        """Parameter temperature_settling"""

        self.temperature_state: Parameter = self.add_parameter(
            "temperature_state",
            label="Temperature tracking state",
            val_mapping={
                "tracking": 2,
                "stable": 1,
                "near": 5,
                "chasing": 6,
                "pot operation": 7,
                "standby": 10,
                "diagnostic": 13,
                "impedance control error": 14,
                "failure": 15,
            },
            get_parser=partial(DynaCool._pick_one, 2, int),
            get_cmd="TEMP?",
        )
        """Parameter temperature_state"""

        self.field_measured: Parameter = self.add_parameter(
            "field_measured",
            label="Field",
            unit="T",
            get_cmd=self._measured_field_getter,
        )
        """Parameter field_measured"""

        self.field_target: Parameter = self.add_parameter(
            "field_target",
            label="Field target",
            unit="T",
            get_cmd=None,
            set_cmd=None,
            vals=vals.Numbers(-14, 14),
        )
        """Parameter field_target"""

        self.field_ramp: Parameter = self.add_parameter(
            "field_ramp",
            label="Field [ramp]",
            unit="T",
            get_cmd=None,
            set_cmd=self._field_ramp_setter,
            vals=vals.Numbers(-14, 14),
        )
        """Parameter field_ramp"""

        self.field_rate: Parameter = self.add_parameter(
            "field_rate",
            label="Field rate",
            unit="T/s",
            get_parser=lambda x: x * 1e-4,  # Oe to T
            set_parser=lambda x: x * 1e4,  # T to Oe
            set_cmd=None,
            get_cmd=None,
            initial_value=0,
            vals=vals.Numbers(0, 1),
        )
        """Parameter field_rate"""

        self.field_approach: Parameter = self.add_parameter(
            "field_approach",
            label="Field ramp approach",
            val_mapping={"linear": 0, "no overshoot": 1, "oscillate": 2},
            set_cmd=None,
            get_cmd=None,
            initial_value="linear",
        )
        """Parameter field_approach"""

        self.magnet_state: Parameter = self.add_parameter(
            "magnet_state",
            label="Magnet state",
            val_mapping={
                "unknown": 0,
                "stable": 1,
                "switch warming": 2,
                "switch cool": 3,
                "holding": 4,
                "iterate": 5,
                "ramping": 6,
                "ramping ": 7,  # map must have inverse
                "resetting": 8,
                "current error": 9,
                "switch error": 10,
                "quenching": 11,
                "charging error": 12,
                "power supply error": 14,
                "failure": 15,
            },
            get_parser=partial(DynaCool._pick_one, 2, int),
            get_cmd="FELD?",
        )
        """Parameter magnet_state"""

        self.chamber_temperature: Parameter = self.add_parameter(
            "chamber_temperature",
            label="Chamber Temperature",
            unit="K",
            get_parser=partial(DynaCool._pick_one, 1, float),
            get_cmd="CHAT?",
        )
        """Parameter chamber_temperature"""

        self.chamber_state: Parameter = self.add_parameter(
            "chamber_state",
            label="Chamber vacuum state",
            val_mapping={
                "purged and sealed": 1,
                "vented and sealed": 2,
                "sealed": 3,
                "performing purge/seal": 4,
                "performing vent/seal": 5,
                "pre-high vacuum": 6,
                "high vacuum": 7,
                "pumping continuously": 8,
                "flooding continuously": 9,
            },
            get_parser=partial(DynaCool._pick_one, 1, int),
            get_cmd="CHAM?",
        )
        """Parameter chamber_state"""

        self.field_tolerance: Parameter = self.add_parameter(
            "field_tolerance",
            label="Field Tolerance",
            unit="T",
            get_cmd=None,
            set_cmd=None,
            vals=vals.Numbers(0, 1e-2),
            set_parser=float,
            docstring="The tolerance below which fields are "
            "considered identical in a "
            "blocking ramp.",
            initial_value=5e-4,
        )
        """The tolerance below which fields are considered identical in a blocking ramp."""

        # The error code of the latest command
        self._error_code = 0

        # we must know all parameter values because of interlinked parameters
        self.snapshot(update=True)

        # it is a safe default to set the target to the current value
        self.field_target(self.field_measured())

        self.connect_message()

    @property
    def error_code(self) -> int:
        return self._error_code

    @staticmethod
    def _pick_one(which_one: int, parser: type, resp: str) -> Any:
        """
        Since most of the API calls return several values in a comma-separated
        string, here's a convenience function to pick out the substring of
        interest
        """
        return parser(resp.split(', ')[which_one])

    def get_idn(self) -> dict[str, Optional[str]]:
        response = self.ask('*IDN?')
        # just clip out the error code
        id_parts = response[2:].split(', ')

        return dict(zip(('vendor', 'model', 'serial', 'firmware'), id_parts))

    def ramp(self, mode: str = "blocking") -> None:
        """
        Ramp the field to the value given by the `field_target` parameter

        Args:
            mode: how to ramp, either "blocking" or "non-blocking". In
                "blocking" mode, this function does not return until the
                target field has been reached. In "non-blocking" mode, this
                function immediately returns.
        """
        if mode not in ['blocking', 'non-blocking']:
            raise ValueError('Invalid ramp mode received. Ramp mode must be '
                             'either "blocking" or "non-blocking", received '
                             f'"{mode}"')

        target_in_tesla = self.field_target()
        # the target must be converted from T to Oersted
        target_in_oe = target_in_tesla*1e4

        start_field = self.field_measured()
        ramp_range = np.abs(start_field - target_in_tesla)
        # as the second argument is zero relative tolerance has no effect.
        if np.allclose([ramp_range], 0, rtol=0, atol=self.field_tolerance()):
            return

        if mode == "blocking":
            self._do_blocking_ramp(target_in_tesla, start_field)
        else:
            self._field_setter(param='field_target',
                               value=target_in_oe)

    def _do_blocking_ramp(self, target_in_tesla: float,
                          start_field_in_tesla: float) -> None:
        """
        Perform a blocking ramp. Only call this function from withing the
        `ramp` method.

        This method is slow; it waits for the magnet to settle. The waiting is
        done in two steps, since users have reported that the magnet state does
        not immediately change to 'ramping' when asked to ramp.
        """

        target_in_oe = target_in_tesla*1e4
        ramp_range = np.abs(target_in_tesla - start_field_in_tesla)

        self._field_setter(param='field_target', value=target_in_oe)

        # step 1: wait for the magnet to actually start ramping
        # NB: depending on the `field_approach`, we may reach the target
        # several times before the ramp is over (oscillations around target)
        while np.abs(self.field_measured() - start_field_in_tesla) \
                < ramp_range * 0.5:
            sleep(self._ramp_time_resolution)

        # step 2: wait for the magnet to report that is has reached the
        # setpoint

        while self.magnet_state() != 'holding':
            sleep(self._ramp_time_resolution)

    def _field_ramp_setter(self, target: float) -> None:
        """
        set_cmd for the field_ramp parameter
        """
        self.field_target(target)
        self.ramp(mode='blocking')

    def _measured_field_getter(self) -> float:
        resp = self.ask('FELD?')
        number_in_oersted = cast(float, DynaCool._pick_one(1, float, resp))
        number_in_tesla = number_in_oersted*1e-4
        return number_in_tesla

    def _field_getter(
        self, param_name: Literal["field_target", "field_rate", "field_approach"]
    ) -> Union[int, float]:
        """
        The combined get function for the three field parameters,
        field_setpoint, field_rate, and field_approach
        """
        raw_response = self.ask('GLFS?')
        sp = self._pick_one(1, float, raw_response)
        rate = self._pick_one(2, float, raw_response)
        approach = self._pick_one(3, int, raw_response)

        return dict(zip(self.field_params, [sp, rate, approach]))[param_name]

    def _field_setter(
        self,
        param: Literal["field_target", "field_rate", "field_approach"],
        value: float,
    ) -> None:
        """
        The combined set function for the three field parameters,
        field_setpoint, field_rate, and field_approach
        """
        temporary_values = list(self.parameters[p].raw_value
                                for p in self.field_params)
        values = cast(list[Union[int, float]], temporary_values)
        values[self.field_params.index(param)] = value

        self.write(f'FELD {values[0]}, {values[1]}, {values[2]}, 0')

    def _temp_getter(
        self,
        param_name: Literal[
            "temperature_setpoint", "temperature_rate", "temperature_settling"
        ],
    ) -> Union[int, float]:
        """
        This function queries the last temperature setpoint (w. rate and mode)
        from the instrument.
        """
        raw_response = self.ask('GLTS?')
        sp = DynaCool._pick_one(1, float, raw_response)
        rate = DynaCool._pick_one(2, float, raw_response)
        mode = DynaCool._pick_one(3, int, raw_response)

        return dict(zip(self.temp_params, [sp, rate, mode]))[param_name]

    def _temp_setter(
        self,
        param: Literal[
            "temperature_setpoint", "temperature_rate", "temperature_settling"
        ],
        value: float,
    ) -> None:
        """
        The setter function for the temperature parameters. All three are set
        with the same call to the instrument API
        """
        temp_values = list(self.parameters[par].raw_value
                           for par in self.temp_params)
        values = cast(list[Union[int, float]], temp_values)
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
