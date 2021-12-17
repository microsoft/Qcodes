from typing import Any

from . import N52xx

class N5222B(N52xx.PNABase):
    def __init__(self, name: str, address: str, **kwargs: Any):
        """Driver for Keysight PNA N5222B."""
        super().__init__(name, address,
                         min_freq=10e6, max_freq=26.5e9,
                         min_power=-30, max_power=13,
                         nports=4,
                         **kwargs)

        attenuators_options = {'217', '219', '220', '417', '419', '420'}
        options = set(self.get_options())
        if attenuators_options.intersection(options):
            self._set_power_limits(min_power=-95, max_power=13)
