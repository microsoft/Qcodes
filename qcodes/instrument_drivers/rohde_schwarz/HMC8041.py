from .private.HMC804x import _RohdeSchwarzHMC804x

class RohdeSchwarzHMC8041(_RohdeSchwarzHMC804x):
    """
    This is the qcodes driver for the Rohde & Schwarz HMC8041 Power Supply
    """
    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, num_channels=1, **kwargs)
