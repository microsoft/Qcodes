from qcodes import VisaInstrument
from qcodes import Instrument
from qcodes.instrument.channel import InstrumentChannel


class B2962AChannel(InstrumentChannel):
    """

    """
    def __init__(self, parent: Instrument, name: str, chan: int) -> None:
        """
        Args:
            parent (Instrument): The instrument to which the channel is
            attached.
            name (str): The name of the channel
            channum (int): The number of the channel in question (1-2)
        """
        # Sanity Check inputs
        if name not in ['ch1', 'ch2']:
            raise ValueError("Invalid Channel: {}, expected 'ch1' or 'ch2'"
                             .format(name))
        if chan not in [1, 2]:
            raise ValueError("Invalid Channel: {}, expected '1' or '2'"
                             .format(chan))

        super().__init__(parent, name)

        self.add_parameter('source_voltage',
                           label="Channel {} Voltage".format(chan),
                           get_cmd='SOURCE{:d}:VOLT?'.format(chan),
                           get_parser=float,
                           set_cmd='SOURCE{:d}:VOLT {{:.8G}}'.format(chan),
                           unit='V')

        self.add_parameter('source_current',
                           label="Channel {} Current".format(chan),
                           get_cmd='SOURCE{:d}:CURR?'.format(chan),
                           get_parser=float,
                           set_cmd='SOURCE{:d}:CURR {{:.8G}}'.format(chan),
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

        self.add_parameter('resistance',
                           get_cmd='MEAS:RES? (@{:d})'.format(chan),
                           get_parser=float,
                           label='Channel {} Resistance'.format(chan),
                           unit='ohm')

        self.add_parameter('voltage_limit',
                           get_cmd='SENS{:d}:VOLT:PROT?'.format(chan),
                           get_parser=float,
                           set_cmd='SENS{:d}:VOLT:PROT {{:.8G}}'.format(chan),
                           label='Channel {} Voltage Limit'.format(chan),
                           unit='V')

        self.add_parameter('current_limit',
                           get_cmd='SENS{:d}:CURR:PROT?'.format(chan),
                           get_parser=float,
                           set_cmd='SENS{:d}:CURR:PROT {{:.8G}}'.format(chan),
                           label='Channel {} Current Limit',
                           unit='A')

        self.add_parameter('enable',
                           get_cmd='OUTP{:d}?'.format(chan),
                           set_cmd='OUTP{:d} {{:d}}'.format(chan),
                           val_mapping={'on':  1, 'off': 0})

        self.add_parameter('source_mode',
                           get_cmd=':SOUR{:d}:FUNC:MODE?'.format(chan),
                           set_cmd=':SOUR{:d}:FUNC:MODE {{:s}}'.format(chan),
                           val_mapping={'current': 'CURR', 'voltage': 'VOLT'})

        self.channel = chan


class B2962A(VisaInstrument):
    """
    This is the qcodes driver for the Keysight B2962A 6.5 Digit Low Noise
    Power Source

    Status: alpha-version.
    TODO:
        - Implement any remaining parameters supported by the device
        - Similar drivers have special handlers to map return values of
          9.9e+37 to inf, is this needed?
    """
    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, terminator='\n', **kwargs)

        # The B2962A supports two channels
        for ch_num in [1, 2]:
            ch_name = "ch{:d}".format(ch_num)
            channel = B2962AChannel(self, ch_name, ch_num)
            self.add_submodule(ch_name, channel)

        self.connect_message()

    def get_idn(self):
        IDN = self.ask_raw('*IDN?')
        vendor, model, serial, firmware = map(str.strip, IDN.split(','))
        IDN = {'vendor': vendor, 'model': model,
               'serial': serial, 'firmware': firmware}
        return IDN
