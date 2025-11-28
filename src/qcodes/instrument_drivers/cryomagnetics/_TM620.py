import re
from time import sleep
from typing import TYPE_CHECKING

from qcodes.instrument import VisaInstrument, VisaInstrumentKWArgs

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from qcodes.parameters import Parameter


class CryomagneticsModelTM620(VisaInstrument):
    """
    Driver for the Cryomagnetics TM 620 temperature monitor.

    Units are kG right now

    Args:
        name: a name for the instrument

        address: VISA address of the device

    """

    float_pattern = re.compile(r"[0-9]+\.[0-9]+")
    default_terminator = "\r\n"

    def __init__(
        self,
        name: str,
        address: str,
        **kwargs: "Unpack[VisaInstrumentKWArgs]",
    ) -> None:
        super().__init__(name, address, **kwargs)

        self.shield: Parameter = self.add_parameter(
            name="shield",
            unit="K",
            get_cmd=self._get_A,
            get_parser=float,
            docstring="55K Shield Temp",
        )
        """55K shield temperature"""

        self.magnet: Parameter = self.add_parameter(
            name="magnet",
            unit="K",
            get_cmd=self._get_B,
            get_parser=float,
            docstring="4K Magnet Temp",
        )
        """4K magnet temperature"""

        self.operating_mode()
        # Communication is a bit slow, so add small time delay to avoid errors
        # during initialization
        sleep(0.1)
        self.connect_message()

    def operating_mode(self, remote: bool = True) -> None:
        """
        Sets the device's operating mode to either remote or local.

        Args:
            remote: If True, sets to remote mode, otherwise sets to local mode.

        """
        if remote:
            self.write("REMOTE")
        else:
            self.write("LOCAL")

    def _get_A(self) -> float:
        """Get 55k shield temperature

        Returns:
            Temperature in Kelvin

        """
        output = self.ask("MEAS? A")
        output = self._parse_output(output)
        numeric_output = self._convert_to_numeric(output)

        return numeric_output

    def _get_B(self) -> float:
        """Get 4k magnet temp

        Returns:
            Temperature in Kelvin

        """
        output = self.ask("MEAS? B")
        output = self._parse_output(output)
        numeric_output = self._convert_to_numeric(output)

        return numeric_output

    def _parse_output(self, output: str) -> str:
        """Extract floating point number from the instrument output string.

        Args:
            output: the string returned from the instrument.

        Returns:
            parsed string containing extracted floating point number.

        """

        match = self.float_pattern.search(output)

        if match:
            return match.group(0)

        self.log.error(f"No floating point number found in output: '{output}'")
        raise ValueError(f"No floating point number found in output: '{output}'")

    def _convert_to_numeric(self, raw_value: str) -> float:
        """
        Convert a raw string value to a numeric float.

        Args:
            raw_value: The raw string value to convert.

        Returns:
            The converted float value.

        """
        try:
            numeric_value = float(raw_value)
            return numeric_value
        except ValueError:
            self.log.error(f"Error converting '{raw_value}' to float")
            raise ValueError(f"Unable to convert '{raw_value}' to float")
