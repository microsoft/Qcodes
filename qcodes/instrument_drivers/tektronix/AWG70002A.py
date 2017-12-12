from .AWG70000A import AWG70000A


class AWG70002A(AWG70000A):
    """
    The QCoDeS driver for Tektronix AWG70002A series AWG's.

    All the actual driver meat is in the superclass AWG70000A.
    """

    def __init__(self, name: str, address: str,
                 timeout: float=10, **kwargs) -> None:
        """
        Args:
            name: The name used internally by QCoDeS in the DataSet
            address: The VISA resource name of the instrument
            timeout: The VISA timeout time (in seconds).
        """

        super().__init__(name, address, num_channels=2,
                         timeout=timeout, **kwargs)
