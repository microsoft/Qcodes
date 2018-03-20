# QCoDeS driver for the Keysight 34465A Digital Multimeter

from functools import partial
import numpy as np
import logging

from qcodes.instrument.parameter import ArrayParameter
import qcodes.utils.validators as vals
from qcodes import VisaInstrument
from pyvisa import VisaIOError

log = logging.getLogger(__name__)


class ArrayMeasurement(ArrayParameter):
    """
    Class to return several values. Really represents a measurement routine.
    """

    def __init__(self, name, shape=(1,), *args, **kwargs):

        super().__init__(name, shape=shape, *args, **kwargs)

        self.label = ''
        self.unit = ''
        self.properly_prepared = False

    def prepare(self):
        """
        Prepare the measurement, create the setpoints.

        There is some randomness in the measurement times.
        """

        inst = self._instrument

        N = inst.sample_count()

        # ensure correct instrument settings
        inst.aperture_mode('OFF')  # aperture mode seems slower ON than OFF
        inst.trigger_count(1)
        inst.trigger_delay(0)
        inst.sample_count_pretrigger(0)
        inst.sample_source('TIM')
        inst.autorange('OFF')

        if inst.trigger_source() is None:
            raise ValueError('Trigger source unspecified! Please set '
                             "trigger_source to 'INT' or 'EXT'.")

        # Final step
        self.time_per_point = inst.sample_timer_minimum()
        inst.sample_timer(self.time_per_point)

        self.setpoints = (tuple(np.linspace(0, N*self.time_per_point, N)),)
        self.shape = (N,)

        self.properly_prepared = True

    def get_raw(self):

        if not self.properly_prepared:
            raise ValueError('ArrayMeasurement not properly_prepared. '
                             'Please run prepare().')

        N = self._instrument.sample_count()
        log.debug("Acquiring {} samples.".format(N))

        # Ensure that the measurement doesn't time out
        # TODO (WilliamHPNielsen): What if we wait really long for a trigger?
        old_timeout = self._instrument.visa_handle.timeout
        self._instrument.visa_handle.timeout = N*1000*1.2*self.time_per_point
        self._instrument.visa_handle.timeout += old_timeout

        # Turn off the display to increase measurement speed
        self._instrument.display_text('Acquiring {} samples'.format(N))

        self._instrument.init_measurement()
        try:
            rawvals = self._instrument.ask('FETCH?')
        except VisaIOError:
            rawvals = None
            log.error('Could not pull data from DMM. Perhaps no trigger?')

        self._instrument.visa_handle.timeout = old_timeout

        # parse the acquired values
        try:
            numvals = np.array(list(map(float, rawvals.split(','))))
        except AttributeError:
            numvals = None

        self._instrument.display_clear()

        return numvals


class Keysight_34465A(VisaInstrument):
    """
    Instrument class for Keysight 34465A.


    The driver is written such that usage with models
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
            res_factors['34465A'] = [30e-6, 15e-6, 6e-6] + res_factors['34465A']
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

        def errorparser(rawmssg: str) -> (int, str):
            """
            Parses the error message.

            Args:
                rawmssg: The raw return value of 'SYSTem:ERRor?'

            Returns:
                The error code and the error message.
            """
            code = int(rawmssg.split(',')[0])
            mssg = rawmssg.split(',')[1].strip().replace('"', '')

            return code, mssg

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
                           get_cmd=self._get_voltage,
                           label='Voltage',
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
                           set_cmd='SENSe:VOLTage:DC:RANGe:AUTO {}',
                           get_cmd='SENSe:VOLTage:DC:RANGe:AUTO?',
                           val_mapping={'ON': 1, 'OFF': 0},
                           vals=vals.Enum('ON', 'OFF'))

        self.add_parameter('display_text',
                           label='Display text',
                           set_cmd='DISPLAY:TEXT "{}"',
                           get_cmd='DISPLAY:TEXT?',
                           vals=vals.Strings())

        # TRIGGERING

        self.add_parameter('trigger_count',
                           label='Trigger Count',
                           set_cmd='TRIGger:COUNt {}',
                           get_cmd='TRIGger:COUNt?',
                           get_parser=float,
                           vals=vals.MultiType(vals.Numbers(1, 1e6),
                                               vals.Enum('MIN', 'MAX', 'DEF',
                                                         'INF')))

        self.add_parameter('trigger_delay',
                           label='Trigger Delay',
                           set_cmd='TRIGger:DELay {}',
                           get_cmd='TRIGger:DELay?',
                           vals=vals.MultiType(vals.Numbers(0, 3600),
                                               vals.Enum('MIN', 'MAX', 'DEF')),
                           get_parser=float)

        self.add_parameter('trigger_slope',
                           label='Trigger Slope',
                           set_cmd='TRIGger:SLOPe {}',
                           get_cmd='TRIGger:SLOPe?',
                           vals=vals.Enum('POS', 'NEG'))

        self.add_parameter('trigger_source',
                           label='Trigger Source',
                           set_cmd='TRIGger:SOURce {}',
                           get_cmd='TRIGger:SOURce?',
                           vals=vals.Enum('IMM', 'EXT', 'BUS', 'INT'))

        # SAMPLING

        self.add_parameter('sample_count',
                           label='Sample Count',
                           set_cmd=partial(self._set_databuffer_setpoints,
                                           'SAMPle:COUNt {}'),
                           get_cmd='SAMPle:COUNt?',
                           vals=vals.MultiType(vals.Numbers(1, 1e6),
                                               vals.Enum('MIN', 'MAX', 'DEF')),
                           get_parser=int)

        if DIG:
            self.add_parameter('sample_count_pretrigger',
                               label='Sample Pretrigger Count',
                               set_cmd='SAMPle:COUNt:PRETrigger {}',
                               get_cmd='SAMPle:COUNt:PRETrigger?',
                               vals=vals.MultiType(vals.Numbers(0, 2e6-1),
                                                   vals.Enum('MIN', 'MAX', 'DEF')),
                               get_parser=int,
                               docstring=('Allows collection of the data '
                                          'being digitized the trigger. Reserves '
                                          'memory for pretrigger samples up to the'
                                          ' specified num. of pretrigger samples.')
                               )

        self.add_parameter('sample_source',
                           label='Sample Timing Source',
                           set_cmd='SAMPle:SOURce {}',
                           get_cmd='SAMPle:SOURce?',
                           vals=vals.Enum('IMM', 'TIM'),
                           docstring=('Determines sampling time, immediate'
                                      ' or using sample_timer'))

        self.add_parameter('sample_timer',
                           label='Sample Timer',
                           set_cmd='SAMPle:TIMer {}',
                           get_cmd='SAMPle:TIMer?',
                           unit='s',
                           vals=vals.MultiType(vals.Numbers(0, 3600),
                                               vals.Enum('MIN', 'MAX', 'DEF')),
                           get_parser=float)

        self.add_parameter('sample_timer_minimum',
                           label='Minimal sample time',
                           get_cmd='SAMPle:TIMer? MIN',
                           get_parser=float,
                           unit='s')

        # SYSTEM
        self.add_parameter('error',
                           label='Error message',
                           get_cmd='SYSTem:ERRor?',
                           get_parser=errorparser
                           )

        # The array parameter
        self.add_parameter('data_buffer',
                           parameter_class=ArrayMeasurement)

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
        self.add_function('abort_measurement', call_cmd='ABORt')

        if not silent:
            self.connect_message()

    def _get_voltage(self):
        # TODO: massive improvements!
        # The 'READ?' command will return anything the instrument is set up
        # to return, i.e. not necessarily a voltage (might be current or
        # or resistance) and not necessarily a single value. This function
        # should be aware of the configuration.

        response = self.ask('READ?')

        return float(response)

    def _set_databuffer_setpoints(self, cmd, value):
        """
        set_cmd for all databuffer-setpoint related parameters
        """

        self.data_buffer.properly_prepared = False
        self.write(cmd.format(value))

    def _set_apt_time(self, value):
        self.write('SENSe:VOLTage:DC:APERture {:f}'.format(value))

        # setting aperture time switches aperture mode ON
        self.aperture_mode.get()

    def _set_NPLC(self, value):
        self.write('SENSe:VOLTage:DC:NPLC {:f}'.format(value))

        # This will change data_buffer setpoints (timebase)
        self.data_buffer.properly_prepared = False

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

    def flush_error_queue(self, verbose: bool=True) -> None:
        """
        Clear the instrument error queue.

        Args:
            verbose: If true, the error messages are printed.
                Default: True.
        """

        log.debug('Flushing error queue...')

        err_code, err_message = self.error()
        log.debug('    {}, {}'.format(err_code, err_message))
        if verbose:
            print(err_code, err_message)

        while err_code != 0:
            err_code, err_message = self.error()
            log.debug('    {}, {}'.format(err_code, err_message))
            if verbose:
                print(err_code, err_message)

        log.debug('...flushing complete')
