from typing import Any

from . import N52xx

class N5245A(N52xx.PNAxBase):
    def __init__(self, name: str, address: str, **kwargs: Any):
        super().__init__(name, address,
                         min_freq=10e6, max_freq=50e9,
                         min_power=-30, max_power=13,
                         nports=4,
                         **kwargs)

        attenuators_options = {'219', '419'}
        options = set(self.get_options())
        if attenuators_options.intersection(options):
            self._set_power_limits(min_power=-90, max_power=13)
        if "080" in options:
            self._enable_fom()
