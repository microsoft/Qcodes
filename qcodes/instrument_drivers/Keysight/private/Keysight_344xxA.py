# QCoDeS driver for the Keysight 344xxA Digital Multimeter
import textwrap
from functools import partial
import numpy as np
import logging
from typing import Tuple

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


class _Keysight_344xxA(VisaInstrument):
    """
    Instrument class for Keysight 34460A, 34461A, 34465A and 34470A multimeters.

    Tested with: 34461A, 34465A.

    The driver currently only supports using the instrument as a voltmeter.

    Attributes:
        model (str): The model number of the instrument
        NPLC_list (list): A list of the available Power Line Cycle settings
        ranges (list): A list of the available voltage ranges
    """

    def __init__(self, name, address, silent=False,
                 **kwargs):
        """
        Create an instance of the instrument.

        Args:
            name (str): Name used by QCoDeS. Appears in the DataSet
            address (str): Visa-resolvable instrument address.
            silent (bool): If True, the connect_message of the instrument
                is supressed. Default: False
        Returns:
            _Keysight_344xxA
        """

        super().__init__(name, address, terminator='\n', **kwargs)

        idn = self.IDN.get()
        self.model = idn['model']

        DIG = 'DIG' in self._licenses()

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

        self._resolution_factors = res_factors[self.model]
        self.ranges = ranges[self.model]
        self.NPLC_list = PLCs[self.model]

        ####################################
        # PARAMETERS

        self.add_parameter('line_frequency',
                           get_cmd='SYSTem:LFRequency?',
                           get_parser=int,
                           set_cmd=False,
                           label='Line Frequency',
                           unit='Hz',
                           docstring=('The frequency of the power line where '
                                      'the instrument is plugged')
                           )

        self.add_parameter('NPLC',
                           get_cmd='SENSe:VOLTage:DC:NPLC?',
                           get_parser=float,
                           set_cmd=self._set_NPLC,
                           vals=vals.Enum(*self.NPLC_list),
                           label='Integration time',
                           unit='NPLC',
                           docstring=textwrap.dedent("""\
            Sets the integration time in number of power line cycles (PLC) 
            for DC voltage and ratio measurements. Integration time is the 
            period that the instrument's analog-to-digital (A/D) converter 
            samples the input signal for a measurement. A longer integration 
            time gives better measurement resolution but slower measurement 
            speed.
            
            Only integration times of 1, 10, or 100 PLC provide normal mode 
            (line frequency noise) rejection.
            
            Setting the integration time also sets the measurement 
            resolution.""")
                           )

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

        self.add_parameter('display_enabled',
                           label='Display enabled',
                           set_cmd='DISPlay:STATe {}',
                           get_cmd='DISPlay:STATe?',
                           val_mapping={True: 1, False: 0},
                           docstring=textwrap.dedent("""\
            Disables or enables the front panel display. When disabled, 
            the display dims, and all annunciators are disabled. However, 
            the screen remains on.
            
            Disabling the display improves command execution speed from the 
            remote interface and provides basic security.
            
            Displaying text with `display_text` parameter will work even 
            when the display is disabled.""")
                           )

        self.add_parameter('display_text',
                           label='Display text',
                           set_cmd='DISPLAY:TEXT "{}"',
                           get_cmd='DISPLAY:TEXT?',
                           get_parser=lambda s: s.strip('"'),
                           vals=vals.Strings(),
                           docstring=textwrap.dedent("""\
            Displays the given text on the screen. Specifying empty string 
            moves the display back to its normal state. The same can be 
            achieved by calling `display_clear`.""")
                           )

        self.add_parameter('autozero',
                           label='Autozero',
                           set_cmd='SENSe:VOLTage:DC:ZERO:AUTO {}',
                           get_cmd='SENSe:VOLTage:DC:ZERO:AUTO?',
                           val_mapping={'ON': 1, 'OFF': 0, 'ONCE': 'ONCE'},
                           vals=vals.Enum('ON', 'OFF', 'ONCE'),
                           docstring=textwrap.dedent("""\
            Disables or enables the autozero mode for DC voltage and ratio 
            measurements.
            
            ON:   the DMM internally measures the offset following each 
                  measurement. It then subtracts that measurement from the 
                  preceding reading. This prevents offset voltages present on
                  the DMM’s input circuitry from affecting measurement 
                  accuracy.
            OFF:  the instrument uses the last measured zero measurement and 
                  subtracts it from each measurement. It takes a new zero 
                  measurement each time you change the function, range or 
                  integration time.
            ONCE: the instrument takes one zero measurement and sets 
                  autozero OFF. The zero measurement taken is used for all 
                  subsequent measurements until the next change to the 
                  function, range or integration time. If the specified 
                  integration time is less than 1 PLC, the zero measurement 
                  is taken at 1 PLC to optimize noise rejection. Subsequent 
                  measurements are taken at the specified fast (< 1 PLC) 
                  integration time.""")
                           )

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
                           unit='s',
                           set_cmd='TRIGger:DELay {}',
                           get_cmd='TRIGger:DELay?',
                           vals=vals.MultiType(vals.Numbers(0, 3600),
                                               vals.Enum('MIN', 'MAX', 'DEF')),
                           get_parser=float,
                           docstring="Step size for DC measurements is "
                                     "approximately 1 µs.\nFor AC "
                                     "measurements, step size depends on AC "
                                     "bandwidth."
                           )

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
                           get_parser=float,
                           docstring=textwrap.dedent("""\
            The value is rounded by the instrument to the nearest step. For DC 
            measurements, the step size is 1 µs. For AC measurements, 
            it is AC bandwidth dependent.
            
            Special values are: MIN - recommended minimum, MAX - maximum, 
            DEF - default. In order to obtain the actual value of the 
            parameter that gets set when setting it to one of these special 
            values, just call the get method of the parameter, or use 
            corresponding parameters in this driver, 
            like `sample_timer_minimum`.

            Specifying a value that is between the absolute minimum (assumes 
            no range changes) and the recommended minimum value, 
            may generate a timing violation error when making measurements.
            
            Applying a value less than the absolute minimum will generate an 
            error."""))

        self.add_parameter('sample_timer_minimum',
                           label='Minimal recommended sample time',
                           get_cmd='SAMPle:TIMer? MIN',
                           get_parser=float,
                           unit='s',
                           docstring=textwrap.dedent("""\
            This value is measurement dependent. It depends on such things 
            as the integration time, autozero on or off, autorange on or 
            off, and the measurement range. Basically, the minimum is 
            automatically determined by the instrument so that the sample 
            interval is always greater than the sampling time.
            
            Since the minimum value changes depending on configuration, a 
            command order dependency exists. You must completely configure 
            the measurement before setting the sample timer to minimum, 
            or you may generate an error. A complete configuration includes 
            such things as math statistics or scaling.
            
            When using autorange, the minimum value is the recommended value, 
            not the absolute minimum value. With autorange enabled, minimum 
            value is calculated assuming a single range change will occur 
            for every measurement (not multiple ranges, just one range up or 
            down per measurement)."""))

        # The array parameter
        self.add_parameter('data_buffer',
                           parameter_class=ArrayMeasurement)

        ####################################
        # Model-specific parameters

        if self.model in ['34465A', '34470A']:
            # Define the extreme aperture time values for the 34465A and 34470A
            utility_freq = self.line_frequency()
            if utility_freq == 50:
                apt_times = {'34465A': [0.3e-3, 2],
                            '34470A': [0.3e-3, 2]}
            elif utility_freq == 60:
                apt_times = {'34465A': [0.3e-3, 1.67],
                            '34470A': [0.3e-3, 1.67]}
            if DIG:
                apt_times['34465A'][0] = 20e-6
                apt_times['34470A'][0] = 20e-6

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
                               vals=vals.Numbers(*apt_times[self.model]),
                               docstring=('Setting the aperture time '
                                          'automatically enables the aperture'
                                          ' mode.'))

        if not silent:
            self.connect_message()

    def init_measurement(self) -> None:
        """
        Change the state of the triggering system from "idle" to
        "wait-for-trigger", and clear the previous set of measurements from
        reading memory.

        This method is an "overlapped" command. This means that after
        executing it, you can send other commands that do not affect the
        measurements.

        Storing measurements in reading memory with this method is faster than
        sending measurements to the instrument's output buffer using
        `read` method ("READ?" command) (provided you do not `fetch`,
        "FETCh?" command, until done).
        """
        self.write('INIT')

    def reset(self) -> None:
        self.write('*RST')

    def display_clear(self) -> None:
        """
        Clear text from display. Depending on the display being
        enabled/disabled, this either returns to display's normal state or
        leaves it black, respectively.
        """
        self.write('DISPLay:TEXT:CLEar')
        self.display_text.get()  # also update the parameter value

    def abort_measurement(self) -> None:
        """
        Abort a measurement in progress, returning the instrument to the
        trigger idle state.
        """
        self.write('ABORt')

    def _licenses(self):
        licenses_raw = self.ask('SYST:LIC:CAT?')
        licenses_list = [x.strip('"') for x in licenses_raw.split(',')]
        return licenses_list

    def _get_voltage(self):
        # TODO: massive improvements!
        # The 'READ?' command will return anything the instrument is set up
        # to return, i.e. not necessarily a voltage (might be current or
        # or resistance) and not necessarily a single value. This function
        # should be aware of the configuration.

        response = self.ask('READ?')

        return float(response)

    def fetch(self) -> np.array:
        """
        Waits for measurements to complete and copies all available
        measurements to the instrument's output buffer. The readings remain
        in reading memory.

        This query does not erase measurements from the reading memory. You
        can call this method multiple times to retrieve the same data.

        Returns:
            a 1D numpy array of all measured values that are currently in the
            reading memory
        """
        raw_vals: str = self.ask('FETCH?')
        return _raw_vals_to_array(raw_vals)

    def _read(self) -> np.array:
        """
        Starts a new set of measurements, waits for all measurements to
        complete, and transfers all available measurements.

        This method is similar to calling :meth:`init_measurement` followed
        immediately by :meth:`fetch`.

        Returns:
            a 1D numpy array of all measured values
        """
        raw_vals: str = self.ask('READ?')
        return _raw_vals_to_array(raw_vals)

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

    def error(self) -> Tuple[int, str]:
        """
        Return the first error message in the queue.

        Returns:
            The error code and the error message.
        """
        rawmssg = self.ask('SYSTem:ERRor?')
        code = int(rawmssg.split(',')[0])
        mssg = rawmssg.split(',')[1].strip().replace('"', '')

        return code, mssg

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


def _raw_vals_to_array(raw_vals: str) -> np.array:
    """
    Helper function that converts comma-delimited string of floating-point
    values to a numpy 1D array of them. Most data retrieval command of these
    instruments return data in this format.

    Args:
        raw_vals: comma-delimited string of floating-point values

    Returns:
        numpy 1D array of data
    """
    return np.array(list(map(float, raw_vals.split(','))))
