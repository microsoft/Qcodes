from qcodes import VisaInstrument


class N9030B(VisaInstrument):
    """
    Driver for Keysight N9030B PXA signal analyzer.
    """

    def __init__(self, name: str, address: str, **kwargs: Any) -> None:
        super().__init__(name, address, terminator='\n', **kwargs)

        self.connect_message()
