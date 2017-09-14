from qcodes import VisaInstrument, InstrumentChannel
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils.validators import Numbers, Bool, Enum, Nothing, Ints

def float_int(val):
    """
    Parses int that are returned in exponentiated form (i.e. 1E0)
    """
    return int(float(val))

class GS200Exception(Exception):
    pass

class GS200_Monitor(InstrumentChannel):
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
                               get_parser=float_int) 
            self.add_parameter('delay',
                               label='Measurement Delay',
                               unit='ms',
                               vals=Ints(0, 999999),
                               set_cmd=':SENS:DEL {}', 
                               set_parser=int,
                               get_cmd=':SENS:DEL?',
                               get_parser=float_int)
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
        self.write(':SENS 0')
        self._enabled = False
    def on(self):
        self.write(':SENS 1')
        self._enabled = True
    def state(self):
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
        self.visa_handle.read_termination = "\r\n"

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
                           get_parser=self._get_source_mode,
                           vals=Enum('VOLT', 'CURR'))
        self.add_parameter('range',
                           label='Source Range',
                           unit='?', # This will be set by the get/set parser
                           get_cmd=':SOUR:RANG?',
                           set_cmd=':SOUR:RANG {}',
                           get_parser=self._getset_range,
                           set_parser=self._getset_range)

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
                           set_cmd=lambda x:0, get_cmd=lambda:0) 
        self.add_parameter('current',
                           label='Current',
                           unit='I',
                           set_cmd=lambda x:0, get_cmd=lambda:0)

        self.add_parameter('voltage_limit',
                           label='Voltage Protection Limit',
                           unit='V',
                           vals=Ints(1, 30),
                           get_cmd=":SOUR:PROT:VOLT?",
                           set_cmd=":SOUR:PROT:VOLT {}",
                           get_parser=float_int,
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
        # Output functions
        self.add_function('on', call_cmd=self.on)
        self.add_function('off', call_cmd=self.off)

        self.connect_message()

        # Update the source ranges and output state
        # This will query mode and range
        self.output.get()
        self.source_mode.get()

    def on(self):
        self.write('OUTPUT 1')
        self.measure._output = True

    def off(self):
        self.write('OUTPUT 0')
        self.measure._output = False

    def state(self):
        state = int(self.ask('OUTPUT?'))
        self.measure._output = bool(state)
        return state

    def _update_vals(self, source_mode=None, source_range=None):
        # Update source mode
        if source_mode is None:
            source_mode = self.ask(":SOUR:FUNC?")
        # Get source range if auto-range is off
        if source_range is None and not self.auto_range.get():
            source_range = float(self.ask("SOUR:RANG?"))

        # Setup source based on what mode we are in
        # Range is updated if auto-range is off
        if source_mode == 'VOLT':
            self.current._set_set(None, None)
            self.current._set_get(None, None)
            if self.auto_range.get():
                self.voltage._set_set(":SOUR:LEV:AUTO {:.5e}", float)
            else:
                self.voltage._set_set(":SOUR:LEV {:.5e}", float)
            self.voltage._set_get(":SOUR:LEV?", float)

            self.current.set_validator(Nothing("Current cannot be set in voltage mode"))
            if self.auto_range.get():
                self.voltage.set_validator(Numbers(-30, 30))
            else:
                self.voltage.set_validator(Numbers(-source_range, source_range))

            self.range.unit = "V"
        else:
            self.voltage._set_set(None, None)
            self.voltage._set_get(None, None)
            if self.auto_range.get():
                self.current._set_set(":SOUR:LEV:AUTO {:.5e}", float)
            else:
                self.current._set_set(":SOUR:LEV {:.5e}", float)
            self.current._set_get(":SOUR:LEV?", float)

            self.voltage.set_validator(Nothing("Voltage cannot be set in current mode"))
            if self.auto_range.get():
                self.current.set_validator(Numbers(-0.1, 0.1))
            else:
                self.current.set_validator(Numbers(-source_range, source_range))

            self.range.unit = "I"

        # Finally if measurements are enabled, update measurement units
        # Source output is set to false and will be checked when a measurement is made
        if self.measure.present:
            self.measure._update_measurement_enabled(source_mode, source_range, False)

    def _set_auto_range(self, val):
        # Store new autorange setting
        self._auto_range = val
        # Update validators
        self._update_vals()
        # Disable measurement if autorange is on
        if self.measure.present:
            self.measure._enabled &= val

    def _get_source_mode(self, val):
        self._update_vals(source_mode=val)
        return val

    def _set_source_mode(self, val):
        # Cannot set source mode when the output is on
        if self.output.get() == 'on':
            raise GS200Exception("Cannot switch mode while source is on")
        # Write the new mode to the instrument
        self.write("SOUR:FUNC {}".format(val))
        # Update the parameters and validators appropriately
        self._update_vals(source_mode=val)

    def _getset_range(self, val):
        val = float(val)

        # Check appropriate range depending on source mode
        source_mode = self.ask(":SOUR:FUNC?")
        if source_mode == 'VOLT':
            if val not in (10e-3, 100e-3, 1e0, 10e0, 30e0):
                raise ValueError("Invalid voltage range")
        else:
            if val not in (1e-3, 10e-3, 100e-3, 200e-3):
                raise ValueError("Invalid current range")

        # Update validators and parameters
        self._update_vals(source_mode=source_mode, source_range=val)
        return val