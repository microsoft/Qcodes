from typing import TYPE_CHECKING

from typing_extensions import deprecated

from qcodes.instrument import InstrumentBaseKWArgs, InstrumentChannel, VisaInstrument
from qcodes.utils import QCoDeSDeprecationWarning

if TYPE_CHECKING:
    from typing_extensions import Unpack


@deprecated("Unused module", category=QCoDeSDeprecationWarning)
class KeysightSubModule(InstrumentChannel):
    """
    A base class for submodules for the 34980A systems.

    Args:
        parent: the system which the module is installed on
        name: user defined name for the module
        slot: the slot the module is installed
    """

    def __init__(
        self,
        parent: VisaInstrument | InstrumentChannel,
        name: str,
        slot: int,
        **kwargs: "Unpack[InstrumentBaseKWArgs]",
    ) -> None:
        super().__init__(parent, name, **kwargs)

        self.slot = slot


class Keysight34980ASwitchMatrixSubModule(InstrumentChannel):
    def __init__(
        self,
        parent: VisaInstrument | InstrumentChannel,
        name: str,
        slot: int,
        **kwargs: "Unpack[InstrumentBaseKWArgs]",
    ) -> None:
        """
        A base class for **Switch Matrix** submodules for the 34980A systems.

        Args:
            parent: the system which the module is installed on
            name: user defined name for the module
            slot: the slot the module is installed
            **kwargs: Forwarded to base class.
        """
        super().__init__(parent, name, **kwargs)

        self.slot = slot

    def validate_value(self, row: int, column: int) -> None:
        """
        to check if the row and column number is within the range of the module
        layout.

        Args:
            row: row value
            column: column value
        """
        raise NotImplementedError("Please subclass this")

    def to_channel_list(
        self, paths: list[tuple[int, int]], wiring_config: str | None = None
    ) -> str:
        """
        convert the (row, column) pair to a 4-digit channel number 'sxxx', where
        s is the slot number, xxx is generated from the numbering function.
        This may be different for different modules.

        Args:
            paths: list of channels to connect [(r1, c1), (r2, c2), (r3, c3)]
            wiring_config: for 1-wire matrices, values are 'MH', 'ML';
                                 for 2-wire matrices, values are 'M1H', 'M2H',
                                 'M1L', 'M2L'

        Returns:
            in the format of '(@sxxx, sxxx, sxxx, sxxx)', where sxxx is a
            4-digit channel number
        """
        raise NotImplementedError("Please subclass this")

    def is_open(self, row: int, column: int) -> bool:
        """
        to check if a channel is open/disconnected

        Args:
            row: row number
            column: column number

        Returns:
            True if the channel is open/disconnected
            False if it's closed/connected.
        """
        self.validate_value(row, column)
        channel = self.to_channel_list([(row, column)])
        message = self.ask(f"ROUT:OPEN? {channel}")
        return bool(int(message))

    def is_closed(self, row: int, column: int) -> bool:
        """
        to check if a channel is closed/connected

        Args:
            row: row number
            column: column number

        Returns:
            True if the channel is closed/connected
            False if it's open/disconnected.
        """
        self.validate_value(row, column)
        channel = self.to_channel_list([(row, column)])
        message = self.ask(f"ROUT:CLOSe? {channel}")
        return bool(int(message))

    def connect(self, row: int, column: int) -> None:
        """
        to connect/close the specified channels

        Args:
            row: row number
            column: column number
        """
        self.validate_value(row, column)
        channel = self.to_channel_list([(row, column)])
        self.write(f"ROUT:CLOSe {channel}")

    def disconnect(self, row: int, column: int) -> None:
        """
        to disconnect/open the specified channels

        Args:
            row: row number
            column: column number
        """
        self.validate_value(row, column)
        channel = self.to_channel_list([(row, column)])
        self.write(f"ROUT:OPEN {channel}")

    def connect_paths(self, paths: list[tuple[int, int]]) -> None:
        """
        to connect/close the specified channels.

        Args:
            paths: list of channels to connect [(r1, c1), (r2, c2), (r3, c3)]
        """
        for row, column in paths:
            self.validate_value(row, column)
        channel_list_str = self.to_channel_list(paths)
        self.write(f"ROUTe:CLOSe {channel_list_str}")

    def disconnect_paths(self, paths: list[tuple[int, int]]) -> None:
        """
        to disconnect/open the specified channels.

        Args:
            paths: list of channels to connect [(r1, c1), (r2, c2), (r3, c3)]
        """
        for row, column in paths:
            self.validate_value(row, column)
        channel_list_str = self.to_channel_list(paths)
        self.write(f"ROUTe:OPEN {channel_list_str}")

    def are_closed(self, paths: list[tuple[int, int]]) -> list[bool]:
        """
        to check if a list of channels is closed/connected

        Args:
            paths: list of channels [(r1, c1), (r2, c2), (r3, c3)]

        Returns:
            a list of True and/or False
            True if the channel is closed/connected
            False if it's open/disconnected.
        """
        for row, column in paths:
            self.validate_value(row, column)
        channel_list_str = self.to_channel_list(paths)
        messages = self.ask(f"ROUTe:CLOSe? {channel_list_str}")
        return [bool(int(message)) for message in messages.split(",")]

    def are_open(self, paths: list[tuple[int, int]]) -> list[bool]:
        """
        to check if a list of channels is open/disconnected

        Args:
            paths: list of channels [(r1, c1), (r2, c2), (r3, c3)]

        Returns:
            a list of True and/or False
            True if the channel is closed/connected
            False if it's open/disconnected.
        """
        for row, column in paths:
            self.validate_value(row, column)
        channel_list_str = self.to_channel_list(paths)
        messages = self.ask(f"ROUTe:OPEN? {channel_list_str}")
        return [bool(int(message)) for message in messages.split(",")]


@deprecated(
    "Use Keysight34980ASwitchMatrixSubModule", category=QCoDeSDeprecationWarning
)
class KeysightSwitchMatrixSubModule(Keysight34980ASwitchMatrixSubModule):
    pass
