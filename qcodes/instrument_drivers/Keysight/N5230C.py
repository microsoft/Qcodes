from qcodes.instrument_drivers.Keysight.N52xx import N52xxBase


class N5230C(N52xxBase):
    min_freq: float = 300e3
    max_freq: float = 13.5e9
    min_power: float = -90
    max_power: float = 13
    port_count: int = 2

