from functools import partial
from typing import Optional, Union, Any

from qcodes.instrument.parameter import DelegateParameter
from qcodes.instrument.visa import VisaInstrument
from qcodes.instrument.channel import InstrumentChannel
from qcodes.utils.validators import Numbers, Bool, Enum, Ints


def float_round(val: float) -> int:
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
        name: instrument name
        present
    """
    def __init__(self, parent: 'GS200', name: str, present: bool) -> None:
        super().__init__(parent, name)

        self.present = present

        # Start off with all disabled
        self._enabled = False
        self._output = False

        # Set up mode cache. These will be filled in once the parent
        # is fully initialized.
        self._range: Union[None, float] = None
        self._unit: Union[None, str] = None

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

            # Note: Measurement will only run if source and
            # measurement is enabled.
            self.add_parameter('measure',
                               label='<unset>', unit='V/I',
                               get_cmd=self._get_measurement,
                               snapshot_get=False)

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
                               label='Measurement Interval',
                               unit='s',
                               vals=Numbers(0.1, 3600),
                               set_cmd=':SENS:INT {}',
                               set_parser=float,
                               get_cmd=':SENS:INT?',
                               get_parser=float)

    def off(self) -> None:
        """Turn measurement off"""
        self.write(':SENS 0')
        self._enabled = False

    def on(self) -> None:
        """Turn measurement on"""
        self.write(':SENS 1')
        self._enabled = True

    def state(self) -> int:
        """Check measurement state"""
        state = int(self.ask(':SENS?'))
        self._enabled = bool(state)
        return state

    def _get_measurement(self) -> float:
        if self._unit is None or self._range is None:
            raise GS200Exception("Measurement module not initialized.")
        if self._parent.auto_range.get() or (self._unit == 'VOLT'
                                             and self._range < 1):
            # Measurements will not work with autorange, or when
            # range is <1V.
            self._enabled = False
            raise GS200Exception("Measurements will not work when range is <1V"
                                 "or when in auto range mode.")
        if not self._output:
            raise GS200Exception("Output is off.")
        if not self._enabled:
            raise GS200Exception("Measurements are disabled.")
        # If enabled and output is on, then we can perform a measurement.
        return float(self.ask(':MEAS?'))

    def update_measurement_enabled(self, unit: str,
                                   output_range: float) -> None:
        """
        Args:
            unit
            output_range
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


class GS200Program(InstrumentChannel):
    """
    """
    def __init__(self, parent: 'GS200', name: str) -> None:
        super().__init__(parent, name)
        self._repeat = 1
        self._file_name = None

        self.add_parameter("interval",
                           label="the program interval time",
                           unit='s',
                           vals=Numbers(0.1, 3600.0),
                           get_cmd=":PROG:INT?",
                           set_cmd=":PROG:INT {}")

        self.add_parameter("slope",
                           label="the program slope time",
                           unit='s',
                           vals=Numbers(0.1, 3600.0),
                           get_cmd=":PROG:SLOP?",
                           set_cmd=":PROG:SLOP {}")

        self.add_parameter("trigger",
                           label="the program trigger",
                           get_cmd=":PROG:TRIG?",
                           set_cmd=":PROG:TRIG {}",
                           vals=Enum('normal', 'mend'))

        self.add_parameter("save",
                           set_cmd=":PROG:SAVE '{}'",
                           docstring="save the program to the system memory "
                                     "(.csv file)")

        self.add_parameter("load",
                           get_cmd=":PROG:LOAD?",
                           set_cmd=":PROG:LOAD '{}'",
                           docstring="load the program (.csv file) from the "
                                     "system memory")

        self.add_parameter("repeat",
                           label="program execution repetition",
                           get_cmd=":PROG:REP?",
                           set_cmd=":PROG:REP {}",
                           val_mapping={'OFF': 0,
                                        'ON': 1})
        self.add_parameter("count",
                           label="step of the current program",
                           get_cmd=":PROG:COUN?",
                           set_cmd=":PROG:COUN {}",
                           vals=Ints(1, 10000))

        self.add_function('start',
                          call_cmd=":PROG:EDIT:STAR",
                          docstring="start program editing")
        self.add_function('end',
                          call_cmd=":PROG:EDIT:END",
                          docstring="end program editing")
        self.add_function('run',
                          call_cmd=":PROG:RUN",
                          docstring="run the program",)


class GS200(VisaInstrument):
    """
    This is the QCoDeS driver for the Yokogawa GS200 voltage and current source.

    Args:
      name: What this instrument is called locally.
      address: The GPIB or USB address of this instrument
      kwargs: kwargs to be passed to VisaInstrument class
      terminator: read terminator for reads/writes to the instrument.
    """

    def __init__(self, name: str, address: str, terminator: str = "\n",
                 **kwargs: Any) -> None:
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

        # We need to get the source_mode value here as we cannot rely on the
        # default value that may have been changed before we connect to the
        # instrument (in a previous session or via the frontpanel).
        self.source_mode()

        self.add_parameter('voltage_range',
                           label='Voltage Source Range',
                           unit='V',
                           get_cmd=partial(self._get_range, "VOLT"),
                           set_cmd=partial(self._set_range, "VOLT"),
                           vals=Enum(10e-3, 100e-3, 1e0, 10e0, 30e0),
                           snapshot_exclude=self.source_mode() == 'CURR'
                           )

        self.add_parameter('current_range',
                           label='Current Source Range',
                           unit='I',
                           get_cmd=partial(self._get_range, "CURR"),
                           set_cmd=partial(self._set_range, "CURR"),
                           vals=Enum(1e-3, 10e-3, 100e-3, 200e-3),
                           snapshot_exclude=self.source_mode() == "VOLT"
                           )

        self.add_parameter('range',
                           parameter_class=DelegateParameter,
                           source=None
                           )

        # The instrument does not support auto range. The parameter
        # auto_range is introduced to add this capability with
        # setting the initial state at False mode.
        self.add_parameter('auto_range',
                           label='Auto Range',
                           set_cmd=self._set_auto_range,
                           get_cmd=None,
                           initial_cache_value=False,
                           vals=Bool()
                           )

        self.add_parameter('voltage',
                           label='Voltage',
                           unit='V',
                           set_cmd=partial(self._get_set_output, "VOLT"),
                           get_cmd=partial(self._get_set_output, "VOLT"),
                           snapshot_exclude=self.source_mode() == "CURR"
                           )

        self.add_parameter('current',
                           label='Current',
                           unit='I',
                           set_cmd=partial(self._get_set_output, "CURR"),
                           get_cmd=partial(self._get_set_output, "CURR"),
                           snapshot_exclude=self.source_mode() == 'VOLT'
                           )

        self.add_parameter('output_level',
                           parameter_class=DelegateParameter,
                           source=None
                           )

        # We need to pass the source parameter for delegate parameters
        # (range and output_level) here according to the present
        # source_mode.
        if self.source_mode() == 'VOLT':
            self.range.source = self.voltage_range
            self.output_level.source = self.voltage
        else:
            self.range.source = self.current_range
            self.output_level.source = self.current

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
                           val_mapping={'off': 0,
                                        'on': 1})

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

        self.add_submodule('program', GS200Program(self, 'program'))

        self.add_parameter("BNC_out",
                           label="BNC trigger out",
                           get_cmd=":ROUT:BNCO?",
                           set_cmd=":ROUT:BNCO {}",
                           vals=Enum("trigger", "output", "ready"),
                           docstring="Sets or queries the output BNC signal")

        self.add_parameter("BNC_in",
                           label="BNC trigger in",
                           get_cmd=":ROUT:BNCI?",
                           set_cmd=":ROUT:BNCI {}",
                           vals=Enum("trigger", "output"),
                           docstring="Sets or queries the input BNC signal")

        self.add_parameter(
            "system_errors",
            get_cmd=":SYSTem:ERRor?",
            docstring="returns the oldest unread error message from the event "
                      "log and removes it from the log."
        )

        self.connect_message()

    def on(self) -> None:
        """Turn output on"""
        self.write('OUTPUT 1')
        self.measure._output = True

    def off(self) -> None:
        """Turn output off"""
        self.write('OUTPUT 0')
        self.measure._output = False

    def state(self) -> int:
        """Check state"""
        state = int(self.ask('OUTPUT?'))
        self.measure._output = bool(state)
        return state

    def ramp_voltage(self, ramp_to: float, step: float, delay: float) -> None:
        """
        Ramp the voltage from the current level to the specified output.

        Args:
            ramp_to: The ramp target in Volt
            step: The ramp steps in Volt
            delay: The time between finishing one step and
                starting another in seconds.
        """
        self._assert_mode("VOLT")
        self._ramp_source(ramp_to, step, delay)

    def ramp_current(self, ramp_to: float, step: float, delay: float) -> None:
        """
        Ramp the current from the current level to the specified output.

        Args:
            ramp_to: The ramp target in Ampere
            step: The ramp steps in Ampere
            delay: The time between finishing one step and starting
                another in seconds.
        """
        self._assert_mode("CURR")
        self._ramp_source(ramp_to, step, delay)

    def _ramp_source(self, ramp_to: float, step: float, delay: float) -> None:
        """
        Ramp the output from the current level to the specified output

        Args:
            ramp_to: The ramp target in volts/amps
            step: The ramp steps in volts/ampere
            delay: The time between finishing one step and
                starting another in seconds.
        """
        saved_step = self.output_level.step
        saved_inter_delay = self.output_level.inter_delay

        self.output_level.step = step
        self.output_level.inter_delay = delay
        self.output_level(ramp_to)

        self.output_level.step = saved_step
        self.output_level.inter_delay = saved_inter_delay

    def _get_set_output(self, mode: str,
                        output_level: Optional[float] = None
                        ) -> Optional[float]:
        """
        Get or set the output level.

        Args:
            mode: "CURR" or "VOLT"
            output_level: If missing, we assume that we are getting the
                current level. Else we are setting it
        """
        self._assert_mode(mode)
        if output_level is not None:
            self._set_output(output_level)
            return None
        return float(self.ask(":SOUR:LEV?"))

    def _set_output(self, output_level: float) -> None:
        """
        Set the output of the instrument.

        Args:
            output_level: output level in Volt or Ampere, depending
                on the current mode.
        """
        auto_enabled = self.auto_range()

        if not auto_enabled:
            self_range = self.range()
            if self_range is None:
                raise RuntimeError("Trying to set output but not in"
                                   " auto mode and range is unknown.")
        else:
            mode = self.source_mode.get_latest()
            if mode == "CURR":
                self_range = 200E-3
            else:
                self_range = 30.0

        # Check we are not trying to set an out of range value
        if self.range() is None or abs(output_level)\
                > abs(self_range):
            # Check that the range hasn't changed
            if not auto_enabled:
                self_range = self.range.get_latest()
                if self_range is None:
                    raise RuntimeError("Trying to set output but not in"
                                       " auto mode and range is unknown.")
            # If we are still out of range, raise a value error
            if abs(output_level) > abs(self_range):
                raise ValueError("Desired output level not in range"
                                 " [-{self_range:.3}, {self_range:.3}]".
                                 format(self_range=self_range))

        if auto_enabled:
            auto_str = ":AUTO"
        else:
            auto_str = ""
        cmd_str = f":SOUR:LEV{auto_str} {output_level:.5e}"
        self.write(cmd_str)

    def _update_measurement_module(self, source_mode: Optional[str] = None,
                                   source_range: Optional[float] = None
                                   ) -> None:
        """
        Update validators/units as source mode/range changes.

        Args:
            source_mode: "CURR" or "VOLT"
            source_range
        """
        if not self.measure.present:
            return

        if source_mode is None:
            source_mode = self.source_mode.get_latest()
        # Get source range if auto-range is off
        if source_range is None and not self.auto_range():
            source_range = self.range()

        self.measure.update_measurement_enabled(source_mode, source_range)

    def _set_auto_range(self, val: bool) -> None:
        """
        Enable/disable auto range.

        Args:
            val: auto range on or off
        """
        self._auto_range = val
        # Disable measurement if auto range is on
        if self.measure.present:
            # Disable the measurement module if auto range is enabled,
            # because the measurement does not work in the
            # 10mV/100mV ranges.
            self.measure._enabled &= not val

    def _assert_mode(self, mode: str) -> None:
        """
        Assert that we are in the correct mode to perform an operation.

        Args:
            mode: "CURR" or "VOLT"
        """
        if self.source_mode.get_latest() != mode:
            raise ValueError("Cannot get/set {} settings while in {} mode".
                             format(mode, self.source_mode.get_latest()))

    def _set_source_mode(self, mode: str) -> None:
        """
        Set output mode and change delegate parameters' source accordingly.
        Also, exclude/include the parameters from snapshot depending on the
        mode. The instrument does not support 'current', 'current_range'
        parameters in "VOLT" mode and 'voltage', 'voltage_range' parameters
        in "CURR" mode.

        Args:
            mode: "CURR" or "VOLT"

        """
        if self.output() == 'on':
            raise GS200Exception("Cannot switch mode while source is on")

        if mode == "VOLT":
            self.range.source = self.voltage_range
            self.output_level.source = self.voltage
            self.voltage_range.snapshot_exclude = False
            self.voltage.snapshot_exclude = False
            self.current_range.snapshot_exclude = True
            self.current.snapshot_exclude = True
        else:
            self.range.source = self.current_range
            self.output_level.source = self.current
            self.voltage_range.snapshot_exclude = True
            self.voltage.snapshot_exclude = True
            self.current_range.snapshot_exclude = False
            self.current.snapshot_exclude = False

        self.write(f"SOUR:FUNC {mode}")
        # We set the cache here since `_update_measurement_module`
        # needs the current value which would otherwise only be set
        # after this method exits
        self.source_mode.cache.set(mode)
        # Update the measurement mode
        self._update_measurement_module(source_mode=mode)

    def _set_range(self, mode: str, output_range: float) -> None:
        """
        Update range

        Args:
            mode: "CURR" or "VOLT"
            output_range: Range to set. For voltage, we have the ranges [10e-3,
                100e-3, 1e0, 10e0, 30e0]. For current, we have the ranges [1e-3,
                10e-3, 100e-3, 200e-3]. If auto_range = False, then setting the
                output can only happen if the set value is smaller than the
                present range.
        """
        self._assert_mode(mode)
        output_range = float(output_range)
        self._update_measurement_module(source_mode=mode,
                                        source_range=output_range)
        self.write(f':SOUR:RANG {output_range}')

    def _get_range(self, mode: str) -> float:
        """
        Query the present range.

        Args:
            mode: "CURR" or "VOLT"

        Returns:
            range: For voltage, we have the ranges [10e-3, 100e-3, 1e0, 10e0,
                30e0]. For current, we have the ranges [1e-3, 10e-3, 100e-3,
                200e-3]. If auto_range = False, then setting the output can only
                happen if the set value is smaller than the present range.
        """
        self._assert_mode(mode)
        return float(self.ask(":SOUR:RANG?"))
