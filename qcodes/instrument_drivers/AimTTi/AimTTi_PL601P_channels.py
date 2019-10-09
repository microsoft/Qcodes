from qcodes import VisaInstrument, validators as vals
from qcodes import InstrumentChannel, ChannelList
from qcodes.utils.helpers import create_on_off_val_mapping


class AimTTiChannel(InstrumentChannel):
    def __init__(self, parent, name, channel, **kwargs) -> None:
        super().__init__(parent, name, **kwargs)
        self.channel = channel

        self.add_parameter('volt',
                           get_cmd=self._get_voltage_value,
                           get_parse=float,
                           set_cmd=f'V{channel} {{}}',
                           label='Voltage',
                           unit='V')

        self.add_parameter('volt_step_size',
                           get_cmd=self._get_voltage_step_size,
                           get_parser=float,
                           set_cmd=f'DELTAV{channel} {{}}',
                           label='Voltage Step Size',
                           unit='V')

        self.add_parameter('increment_volt_by_step_size',
                           set_cmd=f'INCV{channel}')

        self.add_parameter('decrement_volt_by_step_size',
                           set_cmd=f'DECV{channel}')

        self.add_parameter('curr',
                           get_cmd=self._get_current_value,
                           get_parser=float,
                           set_cmd=f'I{channel} {{}}',
                           label='Current',
                           unit='A')

        self.add_parameter('curr_range',
                           get_cmd=f'IRANGE{channel}?',
                           get_parser=int,
                           set_cmd=self._set_current_range,
                           label='Current Range',
                           vals=vals.Enum(1, 2))

        self.add_parameter('curr_step_size',
                           get_cmd=self._get_current_step_size,
                           get_parser=float,
                           set_cmd=f'DELTAI{channel} {{}}',
                           label='Current Step Size',
                           unit='A')

        self.add_parameter('increment_curr_by_step_size',
                           set_cmd=f'INCI{channel}')

        self.add_parameter('decrement_curr_by_step_size',
                           set_cmd=f'DECI{channel}')

        self.add_parameter('output',
                           get_cmd=f'OP{channel}?',
                           get_parser=float,
                           set_cmd=f'OP{channel} {{}}',
                           val_mapping=create_on_off_val_mapping(on_val=1,
                                                                 off_val=0))

        self.add_parameter('save_setup',
                           get_cmd=None,
                           set_cmd=f'SAV{channel} {{}}')

        self.add_parameter('load_setup',
                           get_cmd=f'RCL{channel} {{}}',
                           set_cmd=None)


    def _get_voltage_value(self) -> float:
        channel_id = self.channel
        _voltage = self.ask_raw(f'V{channel_id}?')
        _voltage_split = _voltage.split()
        return float(_voltage_split[1])

    def _get_current_value(self) -> float:
        channel_id = self.channel
        _current = self.ask_raw(f'I{channel_id}?')
        _current_split = _current.split()
        return float(_current_split[1])

    def _get_voltage_step_size(self) -> float:
        channel_id = self.channel
        _voltage_step_size = self.ask_raw(f'DELTAV{channel_id}?')
        _v_step_size_split = _voltage_step_size.split()
        return float(_v_step_size_split[1])

    def _get_current_step_size(self) -> float:
        channel_id = self.channel
        _current_step_size = self.ask_raw(f'DELTAI{channel_id}?')
        _c_step_size_split = _current_step_size.split()
        return float(_c_step_size_split[1])

    def _set_current_range(self, val: int) -> None:
        channel_id = self.channel
        self.output(False)
        self.write(f'IRANGE{channel_id} {val}')

class AimTTi(VisaInstrument):
    """
    This is the QCoDeS driver for the Aim TTi PL-P series power supply.
    Tested with Aim TTi PL601-P.
    """
    def __init__(self, name, address, numChannels=1) -> None:
        super().__init__(name, address, terminator='\n')

        channels = ChannelList(self, "Channels", AimTTiChannel,
                               snapshotable=False)

        channel = AimTTiChannel(self, f'ch{numChannels}', numChannels)
        channels.append(channel)
        self.add_submodule(f'ch{numChannels}', channel)

        channels.lock()
        self.add_submodule('channels', channels)
        self.connect_message()

