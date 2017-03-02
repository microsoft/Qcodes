# QCoDeS driver for the Keysight 34465A Digital Multimeter

import qcodes.utils.validators as vals
from qcodes import VisaInstrument


class Keysight_34465A(VisaInstrument):
    """
    Instrument class for Keysight 34465A.

    The driver is written such that usage which models
    34460A, 34461A, and 34470A should be seamless.

    Tested with: 34465A.

    The driver currently only supports using the instrument as a voltmeter.

    Attributes:
        model (str): The model number of the instrument
        NPLC_list (list): A list of the available Power Line Cycle settings
        ranges (list): A list of the available voltage ranges
    """

    def __init__(self, name, address, DIG=False, utility_freq=50, silent=False,
                 **kwargs):
        """
        Create an instance of the instrument.

        Args:
            name (str): Name used by QCoDeS. Appears in the DataSet
            address (str): Visa-resolvable instrument address.
            utility_freq (int): The local utility frequency in Hz. Default: 50
            DIG (bool): Is the DIG option installed on the instrument?
                Default: False.
            silent (bool): If True, the connect_message of the instrument
                is supressed. Default: False
        Returns:
            Keysight_34465A
        """
        if utility_freq not in [50, 60]:
            raise ValueError('Can not set utility frequency to '
                             '{}. '.format(utility_freq) +
                             'Please enter either 50 Hz or 60 Hz.')

        super().__init__(name, address, terminator='\n', **kwargs)

        idn = self.IDN.get()
        self.model = idn['model']

        ####################################
        # Instrument specifications

        PLCs = {'34460A': [0.02, 0.2, 1, 10, 100],
                '34461A': [0.02, 0.2, 1, 10, 100],
                '34465A': [0.02, 0.06, 0.2, 1, 10, 100],
                '34470A': [0.02, 0.06, 0.2, 1, 10, 100]
                }
        if DIG:
            PLCs['34465A'] = [0.001, 0.002, 0.006] + PLCs['34465A']
            PLCs['34470A'] = [0.001, 0.002, 0.006] + PLCs['34470A']

        ranges = {'34460A': [10**n for n in range(-3, 9)],  # 1 m to 100 M
                  '34461A': [10**n for n in range(-3, 9)],  # 1 m to 100 M
                  '34465A': [10**n for n in range(-3, 10)],  # 1 m to 1 G
                  '34470A': [10**n for n in range(-3, 10)],  # 1 m to 1 G
                  }

        # The resolution factor order matches the order of PLCs
        res_factors = {'34460A': [300e-6, 100e-6, 30e-6, 10e-6, 3e-6],
                       '34461A': [100e-6, 10e-6, 3e-6, 1e-6, 0.3e-6],
                       '34465A': [3e-6, 1.5e-6, 0.7e-6, 0.3e-6, 0.1e-6,
                                  0.03e-6],
                       '34470A': [1e-6, 0.5e-6, 0.3e-6, 0.1e-6, 0.03e-6,
                                  0.01e-6]
                       }
        if DIG:
            res_factors['34465A'] = [30e6, 15e-6, 6e-6] + res_factors['34464A']
            res_factors['34470A'] = [30e-6, 10e-6, 3e-6] + res_factors['34470A']

        # Define the extreme aperture time values for the 34465A and 34470A
        if utility_freq == 50:
            apt_times = {'34465A': [0.3e-3, 2],
                         '34470A': [0.3e-3, 2]}
        if utility_freq == 60:
            apt_times = {'34465A': [0.3e-3, 1.67],
                         '34470A': [0.3e-3, 1.67]}
        if DIG:
            apt_times['34465A'][0] = 20e-6
            apt_times['34470A'][0] = 20e-6

        self._resolution_factors = res_factors[self.model]
        self.ranges = ranges[self.model]
        self.NPLC_list = PLCs[self.model]
        self._apt_times = apt_times[self.model]

        ####################################
        # PARAMETERS

        self.add_parameter('NPLC',
                           get_cmd='SENSe:VOLTage:DC:NPLC?',
                           get_parser=float,
                           set_cmd=self._set_NPLC,
                           vals=vals.Enum(*self.NPLC_list),
                           label='Integration time',
                           unit='NPLC')

        self.add_parameter('volt',
                           get_cmd='READ?',
                           label='Voltage',
                           get_parser=float,
                           unit='V')

        self.add_parameter('range',
                           get_cmd='SENSe:VOLTage:DC:RANGe?',
                           get_parser=float,
                           set_cmd='SENSe:VOLTage:DC:RANGe {:f}',
                           vals=vals.Enum(*self.ranges))

        self.add_parameter('resolution',
                           get_cmd='SENSe:VOLTage:DC:RESolution?',
                           get_parser=float,
                           set_cmd=self._set_resolution,
                           label='Resolution',
                           unit='V')

        self.add_parameter('autorange',
                           label='Autorange',
                           set_cmd='SENSe:VOLtage:DC:RANGe:AUTO {}',
                           get_cmd='SENSe:VOLtage:DC:RANGe:AUTO?',
                           val_mapping={'ON': 1, 'OFF': 0},
                           vals=vals.Enum('ON', 'OFF'))

        self.add_parameter('display_text',
                           label='Display text',
                           set_cmd='DISPLAY:TEXT "{}"',
                           get_cmd='DISPLAY:TEXT?',
                           vals=vals.Strings())

        ####################################
        # Model-specific parameters

        if self.model in ['34465A', '34470A']:

            self.add_parameter('aperture_mode',
                               label='Aperture mode',
                               set_cmd='SENSe:VOLTage:DC:APERture:ENABled {}',
                               get_cmd='SENSe:VOLTage:DC:APERture:ENABled?',
                               val_mapping={'ON': 1, 'OFF': 0},
                               vals=vals.Enum('ON', 'OFF'))

            self.add_parameter('aperture_time',
                               label='Aperture time',
                               set_cmd=self._set_apt_time,
                               get_cmd='SENSe:VOLTage:DC:APERture?',
                               get_parser=float,
                               vals=vals.Numbers(*self._apt_times),
                               docstring=('Setting the aperture time '
                                          'automatically enables the aperture'
                                          ' mode.'))

        self.add_function('init_measurement', call_cmd='INIT')
        self.add_function('reset', call_cmd='*RST')
        self.add_function('display_clear', call_cmd=('DISPLay:TEXT:CLEar'))

        if not silent:
            self.connect_message()

    def _set_apt_time(self, value):
        self.write('SENSe:VOLTage:DC:APERture {:f}'.format(value))

        # setting aperture time switches aperture mode ON
        self.aperture_mode.get()

    def _set_NPLC(self, value):
        self.write('SENSe:VOLTage:DC:NPLC {:f}'.format(value))

        # resolution settings change with NPLC
        self.resolution.get()

        # setting NPLC switches off aperture mode
        if self.model in ['34465A', '34470A']:
            self.aperture_mode.get()

    def _set_range(self, value):
        self.write('SENSe:VOLTage:DC:RANGe {:f}'.format(value))

        # resolution settings change with range

        self.resolution.get()

    def _set_resolution(self, value):
        rang = self.range.get()

        # convert both value*range and the resolution factors
        # to strings with few digits, so we avoid floating point
        # rounding errors.
        res_fac_strs = ['{:.1e}'.format(v * rang)
                        for v in self._resolution_factors]
        if '{:.1e}'.format(value) not in res_fac_strs:
            raise ValueError(
                'Resolution setting {:.1e} ({} at range {}) '
                'does not exist. '
                'Possible values are {}'.format(value, value, rang,
                                                res_fac_strs))

        self.write('VOLT:DC:RES {:.1e}'.format(value))

        # NPLC settings change with resolution
        self.NPLC.get()