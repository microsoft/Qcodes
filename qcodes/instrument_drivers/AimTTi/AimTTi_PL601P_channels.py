from qcodes import VisaInstrument, validators as vals
from qcodes import InstrumentChannel, ChannelList


class AimTtiChannel(InstrumentChannel):
    def __init__(self, name, channel, **kwargs):
        super().__init__(name, **kwargs)

        self.add_parameter('volt',
                           get_cmd=f'V{channel}?',
                           get_parser=float,
                           set_cmd=f'V{channel} {{}}',
                           label='Voltage',
                           unit='V')

        self.add_parameter('curr',
                           get_cmd='',
                           get_parser=float,
                           set_cmd='',
                           label='Current',
                           unit='A')

class AimTti(VisaInstrument):
    """
    This is the QCoDeS driver for the Aim TTi PL-P series power supply.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        channels = ChannelList(self, "Channels", AimTtiChannel,
                               snapshotable=False)

        for channel_number in range(1, 4):
            channel = AimTtiChannel(self, "ch{}".format(channel_number),
                                    channel_number)
            channels.append(channel)

        channels.lock()
        self.add_submodule('channels', channels)

