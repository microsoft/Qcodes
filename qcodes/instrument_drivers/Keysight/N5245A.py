from qcodes.instrument_drivers.Keysight.N52xx import N52xxBase


class N5245A(N52xxBase):
    min_freq: float = 10e6
    max_freq: float = 50e9
    min_power: float = -30
    max_power: float = 13
    port_count: int = 4

