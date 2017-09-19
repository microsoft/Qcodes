from qcodes import VisaInstrument, validators as vals
from qcodes.instrument.channel import ChannelList, InstrumentChannel


class AWGChannel(InstrumentChannel):
    """
    Class to hold a channel of the AWG.
    """

    def __init(args):
        passg



class AWG70000A(VisaInstrument):
    """
    The QCoDeS driver for Tektronix AWG70000A series AWG's.

    So far only tested with AWG70002A, but the AWG70001A should be
    very similar [citation needed].
    """

    def __init__(self, name: str, address: str, timeout: float=10,
                 **kwargs) -> None:
        """
        Args:
            name: The name used internally by QCoDeS in the DataSet
            address: The VISA resource name of the instrument
            timeout: The VISA timeout time (in seconds).
        """

        super().__init__(name, address, timeout=timeout, **kwargs)

        self.connect_message()
