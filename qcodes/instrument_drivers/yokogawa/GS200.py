from functools import partial
from typing import Optional, Union

from qcodes import VisaInstrument, InstrumentChannel
from qcodes.utils.validators import Numbers, Bool, Enum, Ints

def float_round(val):
    """
    Rounds a floating number

    Args:
        val: number to be rounded

    Returns:
        Rounded integer
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

    Args:
        parent (GS200)
        name (str): instrument name
        present (bool):
    """
    def __init__(self, parent: 'GS200', name: str, present: bool) -> None:
        super().__init__(parent, name)

        self.present = present

        # Start off with all disabled
        self._enabled = False
        self._output = False
        # Set up monitoring parameters
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
            raise GS200Exception("Measurements will not work when in auto range mode")
        elif self._unit == "VOLT" and self._range < 1:
            raise GS200Exception("Measurements will not work when range is <1V")
        elif not self._enabled:
            raise GS200Exception("Measurements are disabled")

    def update_measurement_enabled(self, unit: str, output_range: float):
        """
        Args:
            unit (str)
            output_range (float)
        """
        # Recheck measurement state next time we do a measurement
        self._enabled = False

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
      terminator (str): read terminator for reads/writes to the instrument.
    """

    def __init__(self, name: str, address: str, terminator: str="\n",
                 **kwargs) -> None:
        super().__init__(name, address, terminator=terminator, **kwargs)

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

        # When getting the mode internally in the driver, look up the mode as recorded by the _cached_mode property,
        # instead of calling source_mode(). This will prevent frequent VISA calls to the instrument. Calling
        # _set_source_mode will change the chased value.
        self._cached_mode = "VOLT"

        # We want to cache the range value so communication with the instrument only happens when the set the
        # range. Getting the range always returns the cached value. This value is adjusted when calling
        # self._set_range
        self._cached_range_value = None # type: Optional[Union[float,int]]

        self.add_parameter('voltage_range',
                           label='Voltage Source Range',
                           unit='V',
                           get_cmd=partial(self._get_range, "VOLT"),
                           set_cmd=partial(self._set_range, "VOLT"),
                           vals=Enum(10e-3, 100e-3, 1e0, 10e0, 30e0))

        # The driver is initialized in the volt mode. In this mode we cannot
        # get 'current_range'. Hence the snapshot is excluded.
        self.add_parameter('current_range',
                           label='Current Source Range',
                           unit='I',
                           get_cmd=partial(self._get_range, "CURR"),
                           set_cmd=partial(self._set_range, "CURR"),
                           vals=Enum(1e-3, 10e-3, 100e-3, 200e-3),
                           snapshot_exclude=True
                           )

        # This is changed through the source_mode interface
        self.range = self.voltage_range

        self._auto_range = False
        self.add_parameter('auto_range',
                           label='Auto Range',
                           set_cmd=self._set_auto_range,
                           get_cmd=lambda: self._auto_range,
                           vals=Bool())

        self.add_parameter('voltage',
                           label='Voltage',
                           unit='V',
                           set_cmd=partial(self._get_set_output, "VOLT"),
                           get_cmd=partial(self._get_set_output, "VOLT")
                           )
        # Again, at init we are in "VOLT" mode. Hence, exclude the snapshot of
        # 'current' as instrument does not support this parameter in
        # "VOLT" mode.
        self.add_parameter('current',
                           label='Current',
                           unit='I',
                           set_cmd=partial(self._get_set_output, "CURR"),
                           get_cmd=partial(self._get_set_output, "CURR"),
                           snapshot_exclude=True
                           )

        # This is changed through the source_mode interface
        self.output_level = self.voltage

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
        # Note: The guard feature can be used to remove common mode noise.
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

    def ramp_voltage(self, ramp_to: float, step: float, delay: float) -> None:
        """
        Ramp the voltage from the current level to the specified output

        Args:
            ramp_to (float): The ramp target in Volt
            step (float): The ramp steps in Volt
            delay (float): The time between finishing one step and starting another in seconds.
        """
        self._assert_mode("VOLT")
        self._ramp_source(ramp_to, step, delay)

    def ramp_current(self, ramp_to: float, step: float, delay: float) -> None:
        """
        Ramp the current from the current level to the specified output

        Args:
            ramp_to (float): The ramp target in Ampere
            step (float): The ramp steps in Ampere
            delay (float): The time between finishing one step and starting another in seconds.
        """
        self._assert_mode("CURR")
        self._ramp_source(ramp_to, step, delay)

    def _ramp_source(self, ramp_to: float, step: float, delay: float) -> None:
        """
        Ramp the output from the current level to the specified output

        Args:
            ramp_to (float): The ramp target in volts/amps
            step (float): The ramp steps in volts/ampere
            delay (float): The time between finishing one step and starting another in seconds.
        """
        saved_step = self.output_level.step
        saved_inter_delay = self.output_level.inter_delay

        self.output_level.step = step
        self.output_level.inter_delay = delay
        self.output_level(ramp_to)

        self.output_level.step = saved_step
        self.output_level.inter_delay = saved_inter_delay

    def _get_set_output(self, mode: str,
                        output_level: float=None) -> Optional[float]:
        """
        Get or set the output level.

        Args:
            mode (str): "CURR" or "VOLT"
            output_level (float), If missing, we assume that we are getting the current level. Else we are setting it
        """
        self._assert_mode(mode)
        if output_level is not None:
            self._set_output(output_level)
            return None
        else:
            return float(self.ask(":SOUR:LEV?"))

    def _set_output(self, output_level: float) -> None:
        """
        Set the output of the instrument.

        Args:
            output_level (float): output level in Volt or Ampere, depending on the current mode
        """
        auto_enabled = self.auto_range()

        if not auto_enabled:
            self_range = self._cached_range_value
            if self_range is None:
                raise RuntimeError("Trying to set output but not in"
                                   " auto mode and range is unknown.")
        else:
            mode = self._cached_mode
            if mode == "CURR":
                self_range = 200E-3
            else:
                self_range = 30

        # Check we are not trying to set an out of range value
        if self._cached_range_value is None or abs(output_level) > abs(self_range):
            # Check that the range hasn't changed
            if not auto_enabled:
                # Update range
                self.range()
                self_range = self._cached_range_value
                if self_range is None:
                    raise RuntimeError("Trying to set output but not in"
                                       " auto mode and range is unknown.")
            # If we are still out of range, raise a value error
            if abs(output_level) > abs(self_range):
                raise ValueError("Desired output level not in range [-{self_range:.3}, {self_range:.3}]".format(
                    self_range=self_range))

        if auto_enabled:
            auto_str = ":AUTO"
        else:
            auto_str = ""
        cmd_str = ":SOUR:LEV{} {:.5e}".format(auto_str, output_level)
        self.write(cmd_str)

    def _update_measurement_module(self, source_mode: str=None, source_range: float=None) -> None:
        """
        Update validators/units as source mode/range changes

        Args:
            source_mode (str): "CURR" or "VOLT"
            source_range (float):
        """
        if not self.measure.present:
            return

        if source_mode is None:
            source_mode = self._cached_mode
        # Get source range if auto-range is off
        if source_range is None and not self.auto_range():
            source_range = self.range()

        self.measure.update_measurement_enabled(source_mode, source_range)

    def _set_auto_range(self, val: bool) -> None:
        """
        Enable/disable auto range.

        Args:
            val (bool): auto range on or off
        """
        self._auto_range = val
        # Disable measurement if auto range is on
        if self.measure.present:
            # Disable the measurement module if auto range is enabled, because the measurement does not work in the
            # 10mV/100mV ranges
            self.measure._enabled &= not val

    def _assert_mode(self, mode: str, check: bool=True) -> None:
        """
        Assert that we are in the correct mode to perform an operation.
        If check is True, we double check the instrument if this check fails.

        Args:
            mode (str): "CURR" or "VOLT"
        """
        if self._cached_mode != mode:
            raise ValueError("Cannot get/set {} settings while in {} mode".format(mode, self._cached_mode))

    def _set_source_mode(self, mode: str) -> None:
        """
        Set output mode. Also, exclude/include the parameters from snapshot
        depending on the mode. The instrument does not support
        'current', 'current_range' parameters in "VOLT" mode and 'voltage',
        'voltage_range' parameters in "CURR" mode.

        Args:
            mode (str): "CURR" or "VOLT"

        """
        if self.output() == 'on':
            raise GS200Exception("Cannot switch mode while source is on")

        if mode == "VOLT":
            self.range = self.voltage_range
            self.output_level = self.voltage
            self.voltage_range.snapshot_exclude = False
            self.voltage.snapshot_exclude = False
            self.current_range.snapshot_exclude = True
            self.current.snapshot_exclude = True
        else:
            self.range = self.current_range
            self.output_level = self.current
            self.voltage_range.snapshot_exclude = True
            self.voltage.snapshot_exclude = True
            self.current_range.snapshot_exclude = False
            self.current.snapshot_exclude = False

        self.write("SOUR:FUNC {}".format(mode))
        self._cached_mode = mode
        # Update the measurement mode
        self._update_measurement_module(source_mode=mode)

    def _set_range(self, mode: str, output_range: float) -> None:
        """
        Update range

        Args:
            mode (str): "CURR" or "VOLT"
            output_range (float): range to set. For voltage we have the ranges [10e-3, 100e-3, 1e0, 10e0, 30e0]. For current
                            we have the ranges [1e-3, 10e-3, 100e-3, 200e-3]. If auto_range = False then setting the
                            output can only happen if the set value is smaller then the present range.
        """
        self._assert_mode(mode)
        output_range = float(output_range)
        self._update_measurement_module(source_mode=mode, source_range=output_range)
        self._cached_range_value = output_range
        self.write(':SOUR:RANG {}'.format(output_range))

    def _get_range(self, mode: str) -> float:
        """
        Query the present range.
        Note: we do not return the cached value here to ensure snapshots correctly update range. In fact, we update the
        cached value when calling this method.

        Args:
            mode (str): "CURR" or "VOLT"

        Returns:
            range (float): For voltage we have the ranges [10e-3, 100e-3, 1e0, 10e0, 30e0]. For current we have the
                            ranges [1e-3, 10e-3, 100e-3, 200e-3]. If auto_range = False then setting the output can
                            only happen if the set value is smaller then the present range.
        """
        self._assert_mode(mode)
        self._cached_range_value = float(self.ask(":SOUR:RANG?"))
        return self._cached_range_value
