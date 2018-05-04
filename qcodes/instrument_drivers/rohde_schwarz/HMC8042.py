from .private.HMC804x import _RohdeSchwarzHMC804x

class RohdeSchwarzHMC8042(_RohdeSchwarzHMC804x):
    """
    This is the qcodes driver for the Rohde & Schwarz HMC8042 Power Supply
    """
    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, num_channels=2, **kwargs)
