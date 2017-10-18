import logging

from qcodes.instrument.channel import InstrumentChannel
from qcodes import IPInstrument, validators as vals

log = logging.getLogger(__name__)


class HS900XAChannel(InstrumentChannel):
    """
    Class to hold a channel of the synthesizer

    Args:
        parent: The instrument to which the channel should be attached
        name: The name of the channel
        channum: The number of the channel (1-indexed)
    """

    def __init__(self, parent: IPInstrument, name: str, channum: int) -> None:
        super().__init__(parent, name)

        def setfreq(channel, setting):  # Works I think.
            return ':CH{}:{}:{{}}MHz'.format(channel, setting)

        def setpwr(channel, setting):  # Works I think.
            return ':CH{}:{}:{{}}dBm'.format(channel, setting)

        def setphase(channel, setting):  # Works I think.
            return ':CH{}:{}:{{}}deg'.format(channel, setting)

        def RFoutput(channel, setting):  # Check mapping
            return ':CH{}:{}:{{}}'.format(channel, setting)

        def getcmd(channel, setting):  # WORKS FOR ALL, DO NOT CHANGE!!
            return ':CH{}:{}?'.format(channel, setting)

        self.add_parameter('freq',
                           label='Channel {} Frequency'.format(channum),
                           get_cmd=getcmd(channum, 'FREQ'),
                           set_cmd=setfreq(channum, 'FREQ'),
                           get_parser=str,
                           set_parser=float)

        self.add_parameter('ch{}_pwr'.format(channum),
                           label='Channel {} Power'.format(channum),
                           get_cmd=getcmd(channum, 'PWR'),
                           set_cmd=setpwr(channum, 'PWR'),
                           get_parser=str,
                           set_parser=float)

        self.add_parameter('ch{}_phase'.format(channum),
                           label='Channel {} Phase'.format(channum),
                           get_cmd=getcmd(channum, 'PHASE'),
                           set_cmd=setphase(channum, 'PHASE'),
                           get_parser=str,
                           set_parser=float)

        self.add_parameter('ch{}_output'.format(channum),
                           label='Channel {} Output'.format(channum),
                           get_cmd=getcmd(channum, 'PWR:RF'),
                           set_cmd=RFoutput(channum, 'PWR:RF'),
                           get_parser=str,
                           # set_parser=float,
                           val_mapping={1: 'ON', 0: 'OFF'})

        self.add_parameter('ch{}_temp'.format(channum),
                           label='Channel {} Temperature'.format(channum),
                           get_cmd=getcmd(channum, 'TEMP'),
                           get_parser=str)


class HS900XA(IPInstrument):

    """
    QCoDeS driver for Holzworth HS9000 series synthesizers

    Args:
        name: The name used by QCoDeS
        address: IP address of instrument
        port: Port to communicate on
        num_channels: Number of physical channels
        timeout: Tim to wait for response before giving up (s)
        terminator: End-of-message termination character to be stripped
            from messages
        persistent: Whether to leave socket open between send commands
        write_confirmation: Whether there are some responses we need to read.
    """

    def __init__(self, name: str, address: str, port: int,
                 num_channels: int,
                 timeout: float=5, terminator: str='\n',
                 persistent: bool=True, write_confirmation: bool=True,
                 **kwargs) -> None:

        super().__init__(name, address=address, port=port,
                         terminator=terminator, timeout=timeout, **kwargs)

        self._address = address
        self._port = port
        self.num_channels = num_channels
        self._timeout = timeout
        self._terminator = terminator
        self._confirmation = write_confirmation
        # TCP Buffer size. Reset to 1400 in case of error (works
        # with this value)
        self._buffer_size = 128

        for n in range(1, 1 + self.num_channel):
            channel = HS900XAChannel(self, 'ch{}'.format(n), n)
            self.add_submodule('ch{}'.format(n), channel)
