from typing import Any

from .AWG70000A import AWG70000A


class TektronixAWG5208(AWG70000A):
    """
    The QCoDeS driver for Tektronix AWG5208
    """

    def __init__(self, name: str, address: str,
                 timeout: float = 10, **kwargs: Any) -> None:
        """
        Args:
            name: The name used internally by QCoDeS in the DataSet
            address: The VISA resource name of the instrument
            timeout: The VISA timeout time (in seconds).
        """

        super().__init__(name, address, num_channels=8,
                         timeout=timeout, **kwargs)


class AWG5208(TektronixAWG5208):
    """
    Alias with non-conformant name left for backwards compatibility
    """

    pass
