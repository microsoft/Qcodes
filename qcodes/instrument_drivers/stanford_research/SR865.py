from qcodes.instrument_drivers.stanford_research.SR86x import SR86x


class SR865(SR86x):
    def __init(self, name, address, reset=False, **kwargs):
        super().__init__(name, address, max_frequency=4E6, reset=reset, **kwargs)
