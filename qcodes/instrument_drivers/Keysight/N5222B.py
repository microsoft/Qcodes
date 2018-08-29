from qcodes.instrument_drivers.Keysight.N52xx_new import N52xxBase


class N5222B(N52xxBase):
    min_freq: float = 10.E6
    max_freq: float = 26.5E9
    min_power: float = -90
    max_power: float = 12
    number_of_channels: int = 4

    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, **kwargs)
