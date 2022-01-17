from typing import Any

from . import N52xx

#  This is not the same class of Keysight devices but seems to work well...

class P9374A(N52xx.PNAxBase):
    def __init__(self, name: str, address: str, **kwargs: Any):
        super().__init__(name, address,
                         min_freq=300e6, max_freq=20e9,
                         min_power=-43, max_power=20,
                         nports=2,
                         **kwargs)

        options = self.get_options()
