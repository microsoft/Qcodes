from qcodes import Instrument, VisaInstrument, validators as vals
from qcodes.instrument.channel import ChannelList, InstrumentChannel


class AWGChannel(InstrumentChannel):
    """
    Class to hold a channel of the AWG.
    """

    def __init__(self,  parent: Instrument, name: str, channel: int) -> None:
        """
        Args:
            parent: The Instrument instance to which the channel is
                to be attached.
            name: The name used in the DataSet
            channel: The channel number, either 1 or 2.
        """

        super().__init__(parent, name)

        num_channels = self._parent.num_channels

        if channel not in list(range(1, num_channels+1)):
            raise ValueError('Illegal channel value.')


class AWG70000A(VisaInstrument):
    """
    The QCoDeS driver for Tektronix AWG70000A series AWG's.

    The drivers for AWG70001A and AWG70002A should be subclasses of this
    general class.
    """

    def __init__(self, name: str, address: str, num_channels: int,
                 timeout: float=10, **kwargs) -> None:
        """
        Args:
            name: The name used internally by QCoDeS in the DataSet
            address: The VISA resource name of the instrument
            timeout: The VISA timeout time (in seconds).

        """

        super().__init__(name, address, timeout=timeout, **kwargs)

        self.connect_message()
