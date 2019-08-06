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
                           label="Channel {} Voltage".format(chan),
                           get_cmd='SOURCE:VOLT? (@{:d})'.format(chan),
                           get_parser=float,
                           set_cmd='SOURCE:VOLT {{:.8G}}, (@{:d})'.format(chan),
                           unit='V')

        self.add_parameter('source_current',
                           label="Channel {} Current".format(chan),
                           get_cmd='SOURCE:CURR? (@{:d})'.format(chan),
                           get_parser=float,
                           set_cmd='SOURCE:CURR {{:.8G}}, (@{:d})'.format(chan),
                           unit='A')
        self.add_parameter('voltage_limit',
                           get_cmd='SOUR:VOLT:PROT? (@{:d})'.format(chan),
                           get_parser=float,
                           set_cmd='SOUR:VOLT:PROT {{:.8G}}, @({:d})'.format(chan),
                           label='Channel {} Voltage Limit'.format(chan),
                           unit='V')

        self.add_parameter('current_limit',
                           get_cmd='SOUR:CURR:PROT? (@{:d})'.format(chan),
                           get_parser=float,
                           set_cmd='SOUR:CURR:PROT {{:.8G}}, (@{:d})'.format(chan),
                           label='Channel {} Current Limit',
                           unit='A')

        self.add_parameter('voltage',
                           get_cmd='MEAS:VOLT? (@{:d})'.format(chan),
                           get_parser=float,
                           label='Channel {} Voltage'.format(chan),
                           unit='V')

        self.add_parameter('current',
                           get_cmd='MEAS:CURR? (@{:d})'.format(chan),
                           get_parser=float,
                           label='Channel {} Current'.format(chan),
                           unit='A')

        self.add_parameter('enable',
                           get_cmd='OUTP:STAT (@{:d})?'.format(chan),
                           set_cmd='OUTP:STAT {{:d}}, (@{:d})'.format(chan),
                           val_mapping={'on':  1, 'off': 0})

        self.add_parameter('source_mode',
                           get_cmd=':OUTP:PMOD? (@{:d})'.format(chan),
                           set_cmd=':OUTP:PMOD {{:s}}, (@{:d})'.format(chan),
                           val_mapping={'current': 'CURR', 'voltage': 'VOLT'})

        self.channel = chan


class N6705B(VisaInstrument):
    def __init__(self, name, address, **kwargs) -> None:
        super().__init__(name, address, terminator='\n', **kwargs)
        self.channels:  List[N6705BChannel] = []
        for ch_num in [1, 2, 3, 4]:
            ch_name = "ch{:d}".format(ch_num)
            channel = N6705BChannel(self, ch_name, ch_num)
            self.add_submodule(ch_name, channel)
            self.channels.append(channel)

        self.connect_message()

    def get_idn(self) -> Dict[str, Optional[str]]:
        IDN = self.ask_raw('*IDN?')
        vendor, model, serial, firmware = map(str.strip, IDN.split(','))
        IDN = {'vendor': vendor, 'model': model,
               'serial': serial, 'firmware': firmware}
        return IDN
