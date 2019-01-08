import textwrap
import numpy as np

import qcodes.utils.validators as vals
from qcodes import VisaInstrument, InstrumentChannel
from qcodes.instrument_drivers.Keysight.private.error_handling import \
    KeysightErrorQueueMixin


class Trigger(InstrumentChannel):
    """Implements triggering parameters and methods of Keysight 344xxA."""

    def __init__(self, parent: '_Keysight_344xxA', name: str, **kwargs):
        super(Trigger, self).__init__(parent, name, **kwargs)

        if self.parent.is_34465A_34470A:
            _max_trigger_count = 1e9
        else:
            _max_trigger_count = 1e6

        self.add_parameter('count',
                           label='Trigger Count',
                           set_cmd='TRIGger:COUNt {}',
                           get_cmd='TRIGger:COUNt?',
                           get_parser=float,
                           vals=vals.MultiType(
                               vals.Numbers(1, _max_trigger_count),
                               vals.Enum('MIN', 'MAX', 'DEF', 'INF')),
                           docstring=textwrap.dedent("""\
            Selects the number of triggers that are accepted by the 
            instrument before returning to the "idle" trigger state.

            You can use the specified trigger count in conjunction with 
            `sample_count`. In this case, the number of measurements 
            returned is the sample count multiplied by the trigger count.

            A variable trigger count is not available from the front panel. 
            However, when you return to remote control of the instrument, 
            the trigger count returns to the previous value you selected."""))

        self.add_parameter('delay',
                           label='Trigger Delay',
                           unit='s',
                           set_cmd='TRIGger:DELay {}',
                           get_cmd='TRIGger:DELay?',
                           vals=vals.MultiType(vals.Numbers(0, 3600),
                                               vals.Enum('MIN', 'MAX', 'DEF')),
                           get_parser=float,
                           docstring=textwrap.dedent("""\
            Sets the delay between the trigger signal and the first 
            measurement. This may be useful in applications where you want 
            to allow the input to settle before taking a measurement or for 
            pacing a burst of measurements.

            Step size for DC measurements is approximately 1 µs. For AC 
            measurements, step size depends on AC bandwidth.

            Selecting a specific trigger delay disables the automatic 
            trigger delay."""))

        self.add_parameter('auto_delay_enabled',
                           label='Auto Trigger Delay Enabled',
                           set_cmd='TRIGger:DELay:AUTO {}',
                           get_cmd='TRIGger:DELay:AUTO?',
                           get_parser=int,
                           val_mapping={True: 1, False: 0},
                           docstring=textwrap.dedent("""\
            Disables or enables automatic trigger delay. If enabled, 
            the instrument determines the delay based on function, range, 
            and integration time or bandwidth.

            Selecting a specific trigger delay using `trigger.delay` disables 
            the automatic trigger delay."""))

        self.add_parameter('slope',
                           label='Trigger Slope',
                           set_cmd='TRIGger:SLOPe {}',
                           get_cmd='TRIGger:SLOPe?',
                           vals=vals.Enum('POS', 'NEG'))

        if self.parent.is_34465A_34470A and self.parent.has_DIG:
            self.add_parameter('level',
                               label='Trigger Level',
                               unit='V',
                               set_cmd='TRIGger:LEVel {}',
                               get_cmd='TRIGger:LEVel?',
                               get_parser=float,
                               vals=vals.MultiType(
                                   vals.Numbers(-1000, 1000),
                                   vals.Enum('MIN', 'MAX', 'DEF')),
                               docstring=textwrap.dedent("""\
                Sets the level on which a trigger occurs when level 
                triggering is enabled (`trigger.source` set to "INT").

                Note that for 100 mV to 100 V ranges and autorange is off, 
                the trigger level can only be set within ±120% of the 
                range."""))

        _trigger_source_docstring = textwrap.dedent("""\
            Selects the trigger source for measurements.

            IMMediate: The trigger signal is always present. When you place 
                the instrument in the "wait-for-trigger" state, the trigger is 
                issued immediately.

            BUS: The instrument is triggered by `trigger.force` method of this 
                driver once the DMM is in the "wait-for-trigger" state.

            EXTernal: The instrument accepts hardware triggers applied to 
                the rear-panel Ext Trig input and takes the specified number 
                of measurements (`sample_count`), each time a TTL pulse 
                specified by `trigger.slope` is received. If the 
                instrument receives an external trigger before it is ready, 
                it buffers one trigger.""")
        _trigger_source_vals = vals.Enum('IMM', 'EXT', 'BUS')

        if self.parent.is_34465A_34470A and self.parent.has_DIG:
            _trigger_source_vals = vals.Enum('IMM', 'EXT', 'BUS', 'INT')
            # extra empty lines are needed for readability of the docstring
            _trigger_source_docstring += textwrap.dedent("""\


            INTernal: Provides level triggering capability. To trigger on a 
                level on the input signal, select INTernal for the source, 
                and set the level and slope with the `trigger.level` and 
                `trigger.slope` parameters.""")

        self.add_parameter('source',
                           label='Trigger Source',
                           set_cmd='TRIGger:SOURce {}',
                           get_cmd='TRIGger:SOURce?',
                           vals=_trigger_source_vals,
                           docstring=_trigger_source_docstring)

    def force(self) -> None:
        """Triggers the instrument if `trigger.source` is "BUS"."""
        self.write('*TRG')


class Sample(InstrumentChannel):
    """Implements sampling parameters of Keysight 344xxA."""

    def __init__(self, parent: '_Keysight_344xxA', name: str, **kwargs):
        super(Sample, self).__init__(parent, name, **kwargs)

        if self.parent.is_34465A_34470A:
            _max_sample_count = 1e9
        else:
            _max_sample_count = 1e6

        self.add_parameter('count',
                           label='Sample Count',
                           set_cmd='SAMPle:COUNt {}',
                           get_cmd='SAMPle:COUNt?',
                           vals=vals.MultiType(
                               vals.Numbers(1, _max_sample_count),
                               vals.Enum('MIN', 'MAX', 'DEF')),
                           get_parser=int,
                           docstring=textwrap.dedent("""\
            Specifies the number of measurements (samples) the instrument 
            takes per trigger.

            MAX selects 1 billion readings. However, when pretrigger is 
            selected, the maximum is 50,000 readings (without the MEM 
            option) or 2,000,000 readings (with the MEM option)"""))

        if self.parent.has_DIG:
            self.add_parameter('pretrigger_count',
                               label='Sample Pretrigger Count',
                               set_cmd='SAMPle:COUNt:PRETrigger {}',
                               get_cmd='SAMPle:COUNt:PRETrigger?',
                               vals=vals.MultiType(
                                   vals.Numbers(0, 2e6 - 1),
                                   vals.Enum('MIN', 'MAX', 'DEF')),
                               get_parser=int,
                               docstring=textwrap.dedent("""\
                Allows collection of the data being digitized the trigger. 
                Reserves memory for pretrigger samples up to the specified 
                num. of pretrigger samples."""))

        if self.parent.is_34465A_34470A:
            self.add_parameter('source',
                               label='Sample Timing Source',
                               set_cmd='SAMPle:SOURce {}',
                               get_cmd='SAMPle:SOURce?',
                               vals=vals.Enum('IMM', 'TIM'),
                               docstring=('Determines sampling time, '
                                          'immediate or using `sample.timer`'))

        self.add_parameter('timer',
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
            like `sample.timer_minimum`.

            Specifying a value that is between the absolute minimum (assumes 
            no range changes) and the recommended minimum value, 
            may generate a timing violation error when making measurements.

            Applying a value less than the absolute minimum will generate an 
            error."""))

        self.add_parameter('timer_minimum',
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


class Display(InstrumentChannel):
    """Implements interaction with the display of Keysight 344xxA."""

    def __init__(self, parent: '_Keysight_344xxA', name: str, **kwargs):
        super(Display, self).__init__(parent, name, **kwargs)

        self.add_parameter('enabled',
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

            Displaying text with `display.text` parameter will work even 
            when the display is disabled."""))

        self.add_parameter('text',
                           label='Display text',
                           set_cmd='DISPLAY:TEXT "{}"',
                           get_cmd='DISPLAY:TEXT?',
                           get_parser=lambda s: s.strip('"'),
                           vals=vals.Strings(),
                           docstring=textwrap.dedent("""\
            Displays the given text on the screen. Specifying empty string 
            moves the display back to its normal state. The same can be 
            achieved by calling `display.clear`."""))

    def clear(self) -> None:
        """
        Clear text from display. Depending on the display being
        enabled/disabled, this either returns to display's normal state or
        leaves it black, respectively.
        """
        self.write('DISPLay:TEXT:CLEar')
        self.text.get()  # also update the parameter value


class _Keysight_344xxA(KeysightErrorQueueMixin, VisaInstrument):
    """
    Instrument class for Keysight 34460A, 34461A, 34465A and 34470A
    multimeters.

    The driver currently only supports using the instrument as a voltmeter
    for DC measurements.

    This driver makes use of submodules for implementing different
    subsystems of the instrument.

    Attributes:
        model: The model number of the instrument
        NPLC_list: A list of the available Power Line Cycle settings
        ranges: A list of the available voltage ranges
    """

    def __init__(self, name: str, address: str, silent: bool=False,
                 **kwargs):
        """
        Create an instance of the instrument.

        Args:
            name: Name used by QCoDeS. Appears in the DataSet
            address: Visa-resolvable instrument address.
            silent: If True, the connect_message of the instrument
                is supressed. Default: False
        """

        super().__init__(name, address, terminator='\n', **kwargs)

        idn = self.IDN.get()
        self.model = idn['model']

        self.is_34465A_34470A = self.model in ['34465A', '34470A']

        ####################################
        # Instrument specifications

        self.has_DIG = 'DIG' in self._licenses()

        PLCs = {'34460A': [0.02, 0.2, 1, 10, 100],
                '34461A': [0.02, 0.2, 1, 10, 100],
                '34465A': [0.02, 0.06, 0.2, 1, 10, 100],
                '34470A': [0.02, 0.06, 0.2, 1, 10, 100]
                }
        if self.has_DIG:
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
        if self.has_DIG:
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
            resolution."""))

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
                           unit='V',
                           vals=vals.MultiType(
                               vals.Numbers(0),
                               vals.Enum('MIN', 'MAX', 'DEF')),
                           docstring=textwrap.dedent("""\
            Selects the measurement resolution for DC voltage and ratio 
            measurements. The resolution is specified in the same units as the 
            selected measurement function, not in number of digits.
            
            You can also specify MIN (best resolution) or MAX (worst 
            resolution).
            
            To achieve normal mode (line frequency noise) rejection, 
            use a resolution that corresponds to an integration time that is 
            an integral number of power line cycles.
            
            Refer to "Resolution Table" or "Range, Resolution and NPLC" 
            sections of the instrument's manual for the available ranges for 
            the resolution values."""))

        self.add_parameter('autorange',
                           label='Autorange',
                           set_cmd='SENSe:VOLTage:DC:RANGe:AUTO {}',
                           get_cmd='SENSe:VOLTage:DC:RANGe:AUTO?',
                           val_mapping={'ON': 1, 'OFF': 0},
                           vals=vals.Enum('ON', 'OFF'))

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
                  integration time."""))

        ####################################
        # Aperture parameters

        if self.is_34465A_34470A:
            # Define the extreme aperture time values for the 34465A and 34470A
            utility_freq = self.line_frequency()
            if utility_freq == 50:
                apt_times = {'34465A': [0.3e-3, 2],
                            '34470A': [0.3e-3, 2]}
            elif utility_freq == 60:
                apt_times = {'34465A': [0.3e-3, 1.67],
                            '34470A': [0.3e-3, 1.67]}
            if self.has_DIG:
                apt_times['34465A'][0] = 20e-6
                apt_times['34470A'][0] = 20e-6

            self.add_parameter('aperture_mode',
                               label='Aperture mode',
                               set_cmd='SENSe:VOLTage:DC:APERture:ENABled {}',
                               get_cmd='SENSe:VOLTage:DC:APERture:ENABled?',
                               val_mapping={'ON': 1, 'OFF': 0},
                               vals=vals.Enum('ON', 'OFF'),
                               docstring=textwrap.dedent("""\
                Enables the setting of integration time in seconds (called 
                aperture time) for DC voltage measurements. If aperture time 
                mode is disabled (default), the integration time is set in PLC 
                (power-line cycles)."""))

            self.add_parameter('aperture_time',
                               label='Aperture time',
                               set_cmd=self._set_apt_time,
                               get_cmd='SENSe:VOLTage:DC:APERture?',
                               get_parser=float,
                               vals=vals.Numbers(*apt_times[self.model]),
                               docstring=textwrap.dedent("""\
                Specifies the integration time in seconds (called aperture 
                time) with 2 µs resolution for DC voltage measurements.
                
                Use this command for precise control of the DMM's 
                integration time. Use `NPLC` for better power-line noise 
                rejection characteristics (NPLC > 1).

                Setting the aperture time automatically enables the aperture 
                mode."""))

        ####################################
        # Submodules

        self.add_submodule('display', Display(self, 'display'))
        self.add_submodule('trigger', Trigger(self, 'trigger'))
        self.add_submodule('sample', Sample(self, 'sample'))

        ####################################
        # Measuring parameter

        self.add_parameter('volt',
                           get_cmd=self._get_voltage,
                           label='Voltage',
                           unit='V')

        ####################################
        # Connect message

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

    def read(self) -> np.array:
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

    def _set_apt_time(self, value):
        self.write('SENSe:VOLTage:DC:APERture {:f}'.format(value))

        # setting aperture time switches aperture mode ON
        self.aperture_mode.get()

    def _set_NPLC(self, value):
        self.write('SENSe:VOLTage:DC:NPLC {:f}'.format(value))

        # resolution settings change with NPLC
        self.resolution.get()

        # setting NPLC switches off aperture mode
        if self.is_34465A_34470A:
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

    def autorange_once(self) -> None:
        """
        Performs immediate autorange and then turns autoranging off.

        The value of the `range` parameter is also updated.
        """
        self.write('SENSe:VOLTage:DC:RANGe:AUTO ONCE')
        self.range.get()


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
