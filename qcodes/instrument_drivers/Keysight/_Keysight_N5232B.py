from typing import Any

from . import N52xx


class KeysightN5232B(N52xx.PNABase):
    def __init__(self, name: str, address: str, **kwargs: Any):
        super().__init__(
            name,
            address,
            min_freq=300e3,
            max_freq=20e9,
            min_power=-90,
            max_power=13,
            nports=4,
            **kwargs
        )
