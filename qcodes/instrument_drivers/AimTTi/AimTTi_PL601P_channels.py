from qcodes import VisaInstrument, validators as vals
from qcodes import InstrumentChannel, ChannelList


class AimTTiChannel(InstrumentChannel):
    def __init__(self, parent, name, channel, **kwargs):
        super().__init__(parent, name, **kwargs)

        self.add_parameter('volt',
                           get_cmd=f'V{channel}?',
                           #get_parser=float,
                           set_cmd=f'V{channel} {{}}',
                           label='Voltage',
                           unit='V')

        self.add_parameter('curr',
                           get_cmd='',
                           get_parser=float,
                           set_cmd='',
                           label='Current',
                           unit='A')

    def _get_voltage_value(self):
        ch = 1
        _voltage = self.ask(f'V{ch}?')
        print(_voltage)
        #_voltage_split = _voltage.split()
        #return _voltage_split[1]
class AimTTi(VisaInstrument):
    """
    This is the QCoDeS driver for the Aim TTi PL-P series power supply.
    """
    def __init__(self, name, address) -> None:
        super().__init__(name, address, terminator='\n')

        channels = ChannelList(self, "Channels", AimTTiChannel,
                               snapshotable=False)

        numChannels = 1
        channel = AimTTiChannel(self, f'ch{numChannels}', numChannels)
        channels.append(channel)
        self.add_submodule(f'ch{numChannels}', channel)

        channels.lock()
        self.add_submodule('channels', channels)
        self.connect_message()

