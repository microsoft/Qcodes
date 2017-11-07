from functools import partial

from qcodes import VisaInstrument, InstrumentChannel
from qcodes.utils.validators import Numbers, Bool, Enum, Nothing, Ints, Validator


def float_round(val):
    """
    Rounds a floating number represented as a string

    Parameters
    ----------
    val: str

    Returns
    -------
    int
    """
    return round(float(val))


class GS200Exception(Exception):
    pass


class GS200_Monitor(InstrumentChannel):
    """
    Monitor part of the GS200. This is only enabled if it is
    installed in the GS200 (it is an optional extra).

    The units will be automatically updated as required.

    To measure:
    `GS200.measure.measure()`
    """
    def __init__(self, parent, name, present):
        super().__init__(parent, name)

        # Is the feature installed in the instrument
        self.present = present
        # Start off with all disabled
        self._enabled = False
        self._output = False
        # Set up monitoring paramters
        if present:
            self.add_parameter('enabled',
                               label='Measurement Enabled',
                               get_cmd=self.state,
                               set_cmd=lambda x: self.on() if x else self.off(),
                               val_mapping={
                                    'off': 0,
                                    'on': 1,
                               })

            # Note: Measurement will only run if source and measurement is enabled.
            self.add_parameter('measure',
                               label='<unset>', unit='V/I',
                               get_cmd=self._get_measurement)

            self.add_parameter('NPLC',
                               label='NPLC',
                               unit='1/LineFreq',
                               vals=Ints(1, 25),
                               set_cmd=':SENS:NPLC {}',
                               set_parser=int,
                               get_cmd=':SENS:NPLC?',
                               get_parser=float_round)
            self.add_parameter('delay',
                               label='Measurement Delay',
                               unit='ms',
                               vals=Ints(0, 999999),
                               set_cmd=':SENS:DEL {}',
                               set_parser=int,
                               get_cmd=':SENS:DEL?',
                               get_parser=float_round)
            self.add_parameter('trigger',
                               label='Trigger Source',
                               set_cmd=':SENS:TRIG {}',
                               get_cmd=':SENS:TRIG?',
                               val_mapping={
                                    'READY': 'READ',
                                    'READ': 'READ',
                                    'TIMER': 'TIM',
                                    'TIM': 'TIM',
                                    'COMMUNICATE': 'COMM',
                                    'IMMEDIATE': 'IMM',
                                    'IMM': 'IMM'
                               })
            self.add_parameter('interval',
                               label='Measurement Interal',
                               unit='s',
                               vals=Numbers(0.1, 3600),
                               set_cmd=':SENS:INT {}',
                               set_parser=float,
                               get_cmd=':SENS:INT?',
                               get_parser=float)

            self.add_function('on', call_cmd=self.on)
            self.add_function('off', call_cmd=self.off)

    def off(self):
        """Turn measurement off"""
        self.write(':SENS 0')
        self._enabled = False

    def on(self):
        """Turn measurement on"""
        self.write(':SENS 1')
        self._enabled = True

    def state(self):
        """Check measurement state"""
        state = int(self.ask(':SENS?'))
        self._enabled = bool(state)
        return state

    def _get_measurement(self):
        """ Check that measurements are enabled and then take a measurement """
        if not self._enabled or not self._output:
            # Check if the output is on
            self._output = self._output or self._parent.output.get() == 'on'

            if self._parent.auto_range.get() or (self._unit == 'VOLT' and self._range < 1):
                # Measurements will not work with autorange, or when range is <1V
                self._enabled = False
            elif not self._enabled:
                # Otherwise check if measurements are enabled
                self._enabled = (self.enabled.get() == 'on')
        # If enabled and output is on, then we can perform a measurement
        if self._enabled and self._output:
            return float(self.ask(':MEAS?'))
        # Otherwise raise an exception
        elif not self._output:
            raise GS200Exception("Output is off")
        elif self._parent.auto_range.get():
            raise GS200Exception("Measurements will not work when in autorange mode")
        elif self._unit == "VOLT" and self._range < 1:
            raise GS200Exception("Measurements will not work when range is <1V")
        elif not self._enabled:
            raise GS200Exception("Measurements are disabled")

    def _update_measurement_enabled(self, unit, output_range, output):
        # Recheck measurement state next time we do a measurement
        self._enabled = False
        # Update output state
        self._output = output
        # Update units
        self._range = output_range
        self._unit = unit
        if self._unit == 'VOLT':
            self.measure.label = 'Source Current'
            self.measure.unit = 'I'
        else:
            self.measure.label = 'Source Voltage'
            self.measure.unit = 'V'


class GS200RangeValidator(Validator):
    def __init__(self, parent_instrument):
        self._parent_instrument = parent_instrument

    def validate(self, value, context=''):

        mode = self._parent_instrument.source_mode()

        if mode == "CURR":
            valid_values = [1e-3, 10e-3, 100e-3, 200e-3]
        else:
            valid_values = [10e-3, 100e-3, 1e0, 10e0, 30e0]

        if value not in valid_values:
            raise ValueError("{} invalid range for mode {}".format(str(value), mode))

    def __repr__(self):
        return "GS200RangeValidator"


class GS200(VisaInstrument):
    """
    This is the qcodes driver for the Yokogawa GS200 voltage and current source

    Args:
      name (str): What this instrument is called locally.
      address (str): The GPIB address of this instrument
      kwargs (dict): kwargs to be passed to VisaInstrument class
    """

    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, **kwargs)
        self.visa_handle.read_termination = "\n"

        self.add_parameter('output',
                           label='Output State',
                           get_cmd=self.state,
                           set_cmd=lambda x: self.on() if x else self.off(),
                           val_mapping={
                               'off': 0,
                               'on': 1,
                           })

        self.add_parameter('source_mode',
                           label='Source Mode',
                           get_cmd=':SOUR:FUNC?',
                           set_cmd=self._set_source_mode,
                           vals=Enum('VOLT', 'CURR'))

        self.add_parameter('range',
                           label='Source Range',
                           unit='?',  # This will be set by the get/set parser
                           get_cmd=':SOUR:RANG?',
                           set_cmd=':SOUR:RANG {}',
                           vals=GS200RangeValidator(self),
                           set_parser=self._range_set_parser,
                           get_parser=float)

        self._auto_range = False
        self.add_parameter('auto_range',
                           label='Auto Range',
                           set_cmd=self._set_auto_range,
                           get_cmd=lambda: self._auto_range,
                           vals=Bool())

        # Note: Get and set for voltage/current will be updated by once
        # range and mode are known
        self.add_parameter('voltage',
                           label='Voltage',
                           unit='V',
                           set_cmd=partial(self._set_output, "VOLT"),
                           get_cmd=partial(self._get_output, "VOLT")
                           )

        self.add_parameter('current',
                           label='Current',
                           unit='I',
                           set_cmd=partial(self._set_output, "CURR"),
                           get_cmd=partial(self._get_output, "CURR")
                           )

        self.add_parameter('voltage_limit',
                           label='Voltage Protection Limit',
                           unit='V',
                           vals=Ints(1, 30),
                           get_cmd=":SOUR:PROT:VOLT?",
                           set_cmd=":SOUR:PROT:VOLT {}",
                           get_parser=float_round,
                           set_parser=int)

        self.add_parameter('current_limit',
                           label='Current Protection Limit',
                           unit='I',
                           vals=Numbers(1e-3, 200e-3),
                           get_cmd=":SOUR:PROT:CURR?",
                           set_cmd=":SOUR:PROT:CURR {:.3f}",
                           get_parser=float,
                           set_parser=float)

        self.add_parameter('four_wire',
                           label='Four Wire Sensing',
                           get_cmd=':SENS:REM?',
                           set_cmd=':SENS:REM {}',
                           val_mapping={
                              'off': 0,
                              'on': 1,
                           })
        # Note: This feature can be used to remove common mode noise.
        # Read the manual to see if you would like to use it
        self.add_parameter('guard',
                          label='Guard Terminal',
                          get_cmd=':SENS:GUAR?',
                          set_cmd=':SENS:GUAR {}',
                          val_mapping={
                              'off': 0,
                              'on': 1,
                          })

        # Return measured line frequency
        self.add_parameter("line_freq",
                           label='Line Frequency',
                           unit="Hz",
                           get_cmd="SYST:LFR?",
                           get_parser=int)

        # Check if monitor is present, and if so enable measurement
        monitor_present = '/MON' in self.ask("*OPT?")
        measure = GS200_Monitor(self, 'measure', monitor_present)
        self.add_submodule('measure', measure)

        # Reset function
        self.add_function('reset', call_cmd='*RST')
        self.connect_message()

        self.output("off")
        self.source_mode("VOLT")
        self.auto_range(True)

    def on(self):
        """Turn output on"""
        self.write('OUTPUT 1')
        self.measure._output = True

    def off(self):
        """Turn output off"""
        self.write('OUTPUT 0')
        self.measure._output = False

    def state(self):
        """Check state"""
        state = int(self.ask('OUTPUT?'))
        self.measure._output = bool(state)
        return state

    def _update_range_units(self, source_mode=None, source_range=None):
        """
        Update validators/units as source mode/range changes

        Parameters
        ----------
        source_mode: str, ["CURR", "VOLT"]
        source_range: float
        """
        if source_mode is None:
            source_mode = self.source_mode()
        # Get source range if auto-range is off
        if source_range is None and not self.auto_range():
            source_range = self.range()

        # Setup source based on what mode we are in
        # Range is updated if auto-range is off
        if source_mode == 'VOLT':
            self.range.unit = "V"
        else:
            self.range.unit = "I"

        # Finally if measurements are enabled, update measurement units
        # Source output is set to false and will be checked when a measurement is made
        if self.measure.present:
            self.measure._update_measurement_enabled(source_mode, source_range, False)

    def _set_output(self, mode, output_level):
        """
        Set the output of the instrument.

        Parameters
        ----------
        mode: str, ["CURR", "VOLT"]
            The desired output mode. If the current output mode does not match the desired output, an exception is
            raised

        output_level: float
        """
        current_mode = self.source_mode()
        if current_mode != mode:
            raise ValueError("Cannot output {} while in {} mode".format(mode, current_mode))

        auto_enabled = self.auto_range()
        self_range = self.range()

        if not auto_enabled and abs(output_level) > abs(self_range):
            raise ValueError("Desired output level not in range [-{self_range:.3}, {self_range:.3}]".format(
                self_range=self_range))

        auto_str = {True: ":AUTO", False: ""}[auto_enabled]
        cmd_str = ":SOUR:LEV{} {:.5e}".format(auto_str, output_level)
        self.write(cmd_str)

    def _get_output(self, mode):
        """
        Get the output of an instrument

        Parameters
        ----------
        mode: str, ["CURR", "VOLT"]
        """
        current_mode = self.source_mode()
        if current_mode != mode:
            raise ValueError("Cannot measure {} while in {} mode".format(mode, current_mode))

        answer = self.ask(":SOUR:LEV?")
        return float(answer)

    def _set_auto_range(self, val):
        """
        Enable/disable auto range.

        Parameters
        ----------
        val: bool
        """
        self._auto_range = val
        self._update_range_units()
        # Disable measurement if auto range is on
        if self.measure.present:
            self.measure._enabled &= val  # TODO: Sebastian, can you explain what you are doing here?

    def _set_source_mode(self, mode):
        """
        Set output mode and update validators

        Parameters
        ----------
        mode: str, ["CURR", "VOLT"]
        """
        if self.output() == 'on':
            raise GS200Exception("Cannot switch mode while source is on")

        self.write("SOUR:FUNC {}".format(mode))
        self._update_range_units(source_mode=mode)

    def _range_set_parser(self, val):
        """
        Update range and validators

        Parameters
        ----------
        val: float
            The output set range is validated by the GS200RangeValidator
        """
        val = float(val)
        source_mode = self.source_mode()
        self._update_range_units(source_mode=source_mode, source_range=val)
        return val
