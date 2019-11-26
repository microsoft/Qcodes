from qcodes import VisaInstrument
from qcodes import Instrument
from qcodes.instrument.channel import InstrumentChannel
from typing import List, Dict, Optional


class N6705BChannel(InstrumentChannel):
    def __init__(self, parent: Instrument, name: str, chan: int) -> None:
        if chan not in [1, 2, 3, 4]:
            raise ValueError('Invalid channel specified')

        super().__init__(parent, name)

        self.add_parameter('source_voltage',
                           label=f"Channel {chan} Voltage",
                           get_cmd=f'SOURCE:VOLT? (@{chan})',
                           get_parser=float,
                           set_cmd=f'SOURCE:VOLT {{:.8G}}, (@{chan})',
                           unit='V')

        self.add_parameter('source_current',
                           label=f"Channel {chan} Current",
                           get_cmd=f'SOURCE:CURR? (@{chan})',
                           get_parser=float,
                           set_cmd=f'SOURCE:CURR {{:.8G}}, (@{chan})',
                           unit='A')

        self.add_parameter('voltage_limit',
                           get_cmd=f'SOUR:VOLT:PROT? (@{chan})',
                           get_parser=float,
                           set_cmd=f'SOUR:VOLT:PROT {{:.8G}}, @({chan})',
                           label=f'Channel {chan} Voltage Limit',
                           unit='V')

        self.add_parameter('voltage',
                           get_cmd=f'MEAS:VOLT? (@{chan})',
                           get_parser=float,
                           label=f'Channel {chan} Voltage',
                           unit='V')

        self.add_parameter('current',
                           get_cmd=f'MEAS:CURR? (@{chan})',
                           get_parser=float,
                           label=f'Channel {chan} Current',
                           unit='A')

        self.add_parameter('enable',
                           get_cmd=f'OUTP:STAT? (@{chan})',
                           set_cmd=f'OUTP:STAT {{:d}}, (@{chan})',
                           val_mapping={'on':  1, 'off': 0})

        self.add_parameter('source_mode',
                           get_cmd=f':OUTP:PMOD? (@{chan})',
                           set_cmd=f':OUTP:PMOD {{:s}}, (@{chan})',
                           val_mapping={'current': 'CURR', 'voltage': 'VOLT'})

        self.channel = chan
        self.ch_name = name


class N6705B(VisaInstrument):
    def __init__(self, name, address, **kwargs) -> None:
        super().__init__(name, address, terminator='\n', **kwargs)
        self.channels:  List[N6705BChannel] = []
        for ch_num in [1, 2, 3, 4]:
            ch_name = f"ch{ch_num}"
            channel = N6705BChannel(self, ch_name, ch_num)
            self.add_submodule(ch_name, channel)
            self.channels.append(channel)

        self.connect_message()

    def get_idn(self) -> Dict[str, Optional[str]]:
        IDNstr = self.ask_raw('*IDN?')
        vendor, model, serial, firmware = map(str.strip, IDNstr.split(','))
        IDN: Dict[str, Optional[str]] = {'vendor': vendor, 'model': model,
                                         'serial': serial, 'firmware': firmware}
        return IDN
