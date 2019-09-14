from qcodes.instrument_drivers.stanford_research.SR86x import SR86x


class SR865A(SR86x):
    """
    The SR865A instrument is almost equal to the SR865, except for the
    max frequency
    """
    def __init__(self, name: str, address: str,
                 reset: bool = False, **kwargs: str) -> None:
        super().__init__(name, address, max_frequency=4E6,
                         reset=reset, **kwargs)
