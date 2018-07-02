from .private.Keysight_344xxA import _Keysight_344xxA

class Keysight_34470A(_Keysight_344xxA):
    """
    This is the qcodes driver for the Keysight 34470A Multimeter
    """
    def __init__(self, name, address, utility_freq=50, silent=False,
                 **kwargs):
        super().__init__(name, address, utility_freq, silent, **kwargs)
