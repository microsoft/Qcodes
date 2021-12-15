from typing import Any

from . import N52xx

class N5222B(N52xx.PNABase):
    """
    Driver for Keysight PNA N5222B
    """
    def __init__(self, name: str, address: str, **kwargs: Any):
        super().__init__(name, address,
                         min_freq=10e6, max_freq=26.5e9,
                         min_power=-30, max_power=13,
                         nports=4,
                         **kwargs)

        options = self.get_options()
        if "419" in options:
            self._set_power_limits(min_power=-90, max_power=13)
