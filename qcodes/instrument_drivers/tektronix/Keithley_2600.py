import warnings

from qcodes import VisaInstrument
from qcodes.instrument.channel import InstrumentChannel
from qcodes.instrument.base import Instrument
import qcodes.utils.validators as vals


class Keithley_2600(VisaInstrument):
    """
    This is the qcodes driver for the Keithley_2600 Source-Meter series,
    tested with Keithley_2614B

    Status: beta-version.
        TODO:
        - Make a channelised version for the two channels
        - add ramping and such stuff

    """
    def __init__(self, name: str, address: str, channel: str,
                 model: str=None, **kwargs) -> None:
        """
        Args:
            name: Name to use internally in QCoDeS
            address: VISA resource address
            channel: Either 'a' or 'b'
            model: The model type, e.g. '2614B'
        """

        warnings.warn("This Keithley driver is old and will be removed "
                      "from QCoDeS soon. Use Keithley_2600_channels "
                      "instead, it is MUCH better.", UserWarning)

        super().__init__(name, address, terminator='\n', **kwargs)

        self._channel = channel

        model = self.visa_handle.ask('print(localnode.model)')

        knownmodels = ['2601B', '2602B', '2604B', '2611B', '2612B',
                       '2614B', '2635B', '2636B']
        if model not in knownmodels:
            kmstring = ('{}, '*(len(knownmodels)-1)).format(*knownmodels[:-1])
            kmstring += 'and {}.'.format(knownmodels[-1])
            raise ValueError('Unknown model. Known model are: ' +
                             kmstring)

        self.model = model

        vranges = {'2601B': [0.1, 1, 6, 40],
                   '2602B': [0.1, 1, 6, 40],
                   '2604B': [0.1, 1, 6, 40],
                   '2611B': [0.2, 2, 20, 200],
                   '2612B': [0.2, 2, 20, 200],
                   '2614B': [0.2, 2, 20, 200],
                   '2635B': [0.2, 2, 20, 200],
                   '2636B': [0.2, 2, 20, 200]
                   }

        # TODO: In pulsed mode, models 2611B, 2612B, and 2614B
        # actually allow up to 10 A.
        iranges = {'2601B': [100e-9, 1e-6, 10e-6, 100e-6,
                             1e-3, 0.01, 0.1, 1, 3],
                   '2602B': [100e-9, 1e-6, 10e-6, 100e-6,
                             1e-3, 0.01, 0.1, 1, 3],
                   '2604B': [100e-9, 1e-6, 10e-6, 100e-6,
                             1e-3, 0.01, 0.1, 1, 3],
                   '2611B': [100e-9, 1e-6, 10e-6, 100e-6,
                             1e-3, 0.01, 0.1, 1, 1.5],
                   '2612B': [100e-9, 1e-6, 10e-6, 100e-6,
                             1e-3, 0.01, 0.1, 1, 1.5],
                   '2614B': [100e-9, 1e-6, 10e-6, 100e-6,
                             1e-3, 0.01, 0.1, 1, 1.5],
                   '2634B': [1e-9, 10e-9, 100e-9, 1e-6, 10e-6, 100e-6,
                             1e-3, 10e-6, 100e-3, 1, 1.5],
                   '2635B': [1e-9, 10e-9, 100e-9, 1e-6, 10e-6, 100e-6,
                             1e-3, 10e-6, 100e-3, 1, 1.5],
                   '2636B': [1e-9, 10e-9, 100e-9, 1e-6, 10e-6, 100e-6,
                             1e-3, 10e-6, 100e-3, 1, 1.5]}

        self.add_parameter('volt',
                           get_cmd='measure.v()',
                           get_parser=float,
                           set_cmd='source.levelv={:.12f}',
                           label='Voltage',
                           unit='V')
        self.add_parameter('curr',
                           get_cmd='measure.i()',
                           get_parser=float,
                           set_cmd='source.leveli={:.12f}',
                           label='Current',
                           unit='A')
        self.add_parameter('res',
                           get_cmd='measure.r()',
                           get_parser=float,
                           set_cmd=False,
                           label='Resistance',
                           unit='Ohm')
        self.add_parameter('mode',
                           get_cmd='source.func',
                           get_parser=float,
                           set_cmd='source.func={:d}',
                           val_mapping={'current': 0, 'voltage': 1},
                           docstring='Selects the output source.')
        self.add_parameter('output',
                           get_cmd='source.output',
                           get_parser=float,
                           set_cmd='source.output={:d}',
                           val_mapping={'on':  1, 'off': 0})
        self.add_parameter('nplc',
                           label='Number of power line cycles',
                           set_cmd='measure.nplc={:.4f}',
                           get_cmd='measure.nplc',
                           get_parser=float,
                           vals=vals.Numbers(0.001, 25))
        # volt range
        # needs get after set (WilliamHPNielsen): why?
        self.add_parameter('sourcerange_v',
                           label='voltage source range',
                           get_cmd='source.rangev',
                           get_parser=float,
                           set_cmd='source.rangev={:.4f}',
                           unit='V',
                           vals=vals.Enum(*vranges[self.model]))
        self.add_parameter('measurerange_v',
                           label='voltage measure range',
                           get_cmd='measure.rangev',
                           set_cmd='measure.rangev={:.4f}',
                           unit='V',
                           vals=vals.Enum(*vranges[self.model]))
        # current range
        # needs get after set
        self.add_parameter('sourcerange_i',
                           label='current source range',
                           get_cmd='source.rangei',
                           get_parser=float,
                           set_cmd='source.rangei={:.4f}',
                           unit='A',
                           vals=vals.Enum(*iranges[self.model]))

        self.add_parameter('measurerange_i',
                           label='current measure range',
                           get_cmd='measure.rangei',
                           get_parser=float,
                           set_cmd='measure.rangei={:.4f}',
                           unit='A',
                           vals=vals.Enum(*iranges[self.model]))
        # Compliance limit
        self.add_parameter('limitv',
                           get_cmd='source.limitv',
                           get_parser=float,
                           set_cmd='source.limitv={:.4f}',
                           unit='V')
        # Compliance limit
        self.add_parameter('limiti',
                           get_cmd='source.limiti',
                           get_parser=float,
                           set_cmd='source.limiti={:.4f}',
                           unit='A')
        # display
        self.add_parameter('display_settext',
                           set_cmd=self._display_settext,
                           vals=vals.Strings())

        self.connect_message()

    def _display_settext(self, text):
        self.visa_handle.write('display.settext("{}")'.format(text))

    def get_idn(self):
        IDN = self.ask_raw('*IDN?')
        vendor, model, serial, firmware = map(str.strip, IDN.split(','))
        model = model[6:]

        IDN = {'vendor': vendor, 'model': model,
               'serial': serial, 'firmware': firmware}
        return IDN

    def display_clear(self):
        """
        This function clears the display, but also leaves it in user mode
        """
        self.visa_handle.write('display.clear()')

    def display_normal(self):
        """
        Set the display to the default mode
        """
        self.visa_handle.write('display.screen = display.SMUA_SMUB')

    def exit_key(self):
        """
        Get back the normal screen after an error:
        send an EXIT key press event
        """
        self.visa_handle.write('display.sendkey(75)')

    def reset(self):
        """
        Reset instrument to factory defaults
        """
        self.write('reset()')
        # remember to update all the metadata
        self.snapshot(update=True)

    def ask(self, cmd):
        return super().ask('print(smu{:s}.{:s})'.format(self._channel, cmd))

    def write(self, cmd):
        super().write('smu{:s}.{:s}'.format(self._channel, cmd))
