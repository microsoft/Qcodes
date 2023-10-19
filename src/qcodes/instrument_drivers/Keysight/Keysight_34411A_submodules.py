from typing import Any

from .private.Keysight_344xxA_submodules import _Keysight_344xxA


class Keysight34411A(_Keysight_344xxA):
    """
    This is the qcodes driver for the Keysight 34411A Multimeter
    """
    def __init__(self, name: str, address: str, silent: bool = False,
                 **kwargs: Any):
        super().__init__(name, address, silent, **kwargs)


class Keysight_34411A(Keysight34411A):
    """
    Alias for backwards compatibility.
    """
