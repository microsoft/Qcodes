import re
import time
from typing import Any, Sequence, Union

import numpy as np
from pyvisa import errors

from qcodes import (
    ChannelList,
    InstrumentChannel,
    Parameter,
    ParameterWithSetpoints,
    VisaInstrument,
)
from qcodes.instrument.base import _BaseParameter
from qcodes.utils.validators import Arrays, Bool, Enum, Ints, Numbers


class PNAAxisParameter(Parameter):
    def __init__(
        self,
        startparam: Parameter,
        stopparam: Parameter,
        pointsparam: Parameter,
        **kwargs: Any,
    ):
        """
        Axis parameter for traces from the PNA
        """
        super().__init__(**kwargs)

        self._startparam = startparam
        self._stopparam = stopparam
        self._pointsparam = pointsparam

    def get_raw(self) -> np.ndarray:
        """
        Return the axis values, with values retrieved from the parent instrument
        """
        # pylint: disable=line-too-long
        return np.linspace(self._startparam(), self._stopparam(), self._pointsparam())  # type: ignore


class PNALogAxisParamter(PNAAxisParameter):
    def get_raw(self) -> np.ndarray:
        """
        Return the axis values on a log scale, with values retrieved from
        the parent instrument
        """
        # pylint: disable=line-too-long
        return np.geomspace(self._startparam(), self._stopparam(), self._pointsparam())  # type: ignore


class PNATimeAxisParameter(PNAAxisParameter):
    def get_raw(self) -> np.ndarray:
        """
        Return the axis values on a time scale, with values retrieved from
        the parent instrument
        """
        return np.linspace(0, self._stopparam(), self._pointsparam())  # type: ignore


class FormattedSweep(ParameterWithSetpoints):
    """
    Mag will run a sweep, including averaging, before returning data.
    As such, wait time in a loop is not needed.
    """

    def __init__(
        self,
        name: str,
        instrument: "PNABase",
        sweep_format: str,
        label: str,
        unit: str,
        memory: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, instrument=instrument, label=label, unit=unit, **kwargs)
        self.sweep_format = sweep_format
        self.memory = memory

    @property
    def setpoints(self) -> Sequence[_BaseParameter]:
        """
        Overwrite setpoint parameter to ask the PNA what type of sweep
        """
        if self.instrument is None:
            raise RuntimeError(
                "Cannot return setpoints if not attached " "to instrument"
            )
        root_instrument: "PNABase" = self.root_instrument  # type: ignore
        sweep_type = root_instrument.sweep_type()
        if sweep_type == "LIN":
            return (root_instrument.frequency_axis,)
        elif sweep_type == "LOG":
            return (root_instrument.frequency_log_axis,)
        elif sweep_type == "CW":
            return (root_instrument.time_axis,)
        else:
            raise NotImplementedError(f"Axis for type {sweep_type} not implemented yet")

    @setpoints.setter
    def setpoints(self, val: Any) -> None:
        """
        Stub to allow initialization. Ignore any set attempts on setpoint as we
        figure it out on the fly.
        """
        return

    def get_raw(self) -> Sequence[float]:
        if self.instrument is None:
            raise RuntimeError("Cannot get data without instrument")
        root_instr = self.instrument.root_instrument
        # Check if we should run a new sweep
        if root_instr.auto_sweep():
            prev_mode = self.instrument.run_sweep()
        # Ask for data, setting the format to the requested form
        self.instrument.format(self.sweep_format)
        data = root_instr.visa_handle.query_binary_values('CALC:DATA? FDATA',
                                                          datatype='f',
                                                          is_big_endian=True)
        data = np.array(data)
        # Restore previous state if it was changed
        if root_instr.auto_sweep():
            root_instr.sweep_mode(prev_mode)

        return data


class PNAPort(InstrumentChannel):
    """
    Allow operations on individual PNA ports.
    Note: This can be expanded to include a large number of extra parameters...
    """

    def __init__(
        self,
        parent: "PNABase",
        name: str,
        port: int,
        min_power: Union[int, float],
        max_power: Union[int, float],
        **kwargs: Any,
    ) -> None:
        super().__init__(parent, name, **kwargs)

        self.port = int(port)
        if self.port < 1 or self.port > 4:
            raise ValueError("Port must be between 1 and 4.")

        pow_cmd = f"SOUR:POW{self.port}"
        self.add_parameter("source_power",
                           label="power",
                           unit="dBm",
                           get_cmd=f"{pow_cmd}?",
                           set_cmd=f"{pow_cmd} {{}}",
                           get_parser=float,
                           vals=Numbers(min_value=min_power,
                                        max_value=max_power))

    def _set_power_limits(self,
                          min_power: Union[int, float],
                          max_power: Union[int, float]) -> None:
        """
        Set port power limits
        """
        self.source_power.vals = Numbers(min_value=min_power,
                                         max_value=max_power)


class PNATrace(InstrumentChannel):
    """
    Allow operations on individual PNA traces.
    """

    def __init__(
        self,
        parent: "PNABase",
        name: str,
        trace_name: str,
        trace_num: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(parent, name, **kwargs)
        self.trace_name = trace_name
        self.trace_num = trace_num

        # Name of parameter (i.e. S11, S21 ...)
        self.add_parameter('trace',
                           label='Trace',
                           get_cmd=self._Sparam,
                           set_cmd=self._set_Sparam)
        # Format
        # Note: Currently parameters that return complex values are not
        # supported as there isn't really a good way of saving them into the
        # dataset
        self.add_parameter('format',
                           label='Format',
                           get_cmd='CALC:FORM?',
                           set_cmd='CALC:FORM {}',
                           vals=Enum('MLIN', 'MLOG', 'PHAS',
                                     'UPH', 'IMAG', 'REAL'))

        # And a list of individual formats
        self.add_parameter('magnitude',
                           sweep_format='MLOG',
                           label='Magnitude',
                           unit='dB',
                           parameter_class=FormattedSweep,
                           vals=Arrays(shape=(self.parent.points,)))
        self.add_parameter('linear_magnitude',
                           sweep_format='MLIN',
                           label='Magnitude',
                           unit='ratio',
                           parameter_class=FormattedSweep,
                           vals=Arrays(shape=(self.parent.points,)))
        self.add_parameter('phase',
                           sweep_format='PHAS',
                           label='Phase',
                           unit='deg',
                           parameter_class=FormattedSweep,
                           vals=Arrays(shape=(self.parent.points,)))
        self.add_parameter('unwrapped_phase',
                           sweep_format='UPH',
                           label='Phase',
                           unit='deg',
                           parameter_class=FormattedSweep,
                           vals=Arrays(shape=(self.parent.points,)))
        self.add_parameter("group_delay",
                           sweep_format='GDEL',
                           label='Group Delay',
                           unit='s',
                           parameter_class=FormattedSweep,
                           vals=Arrays(shape=(self.parent.points,)))
        self.add_parameter('real',
                           sweep_format='REAL',
                           label='Real',
                           unit='LinMag',
                           parameter_class=FormattedSweep,
                           vals=Arrays(shape=(self.parent.points,)))
        self.add_parameter('imaginary',
                           sweep_format='IMAG',
                           label='Imaginary',
                           unit='LinMag',
                           parameter_class=FormattedSweep,
                           vals=Arrays(shape=(self.parent.points,)))

    def run_sweep(self) -> str:
        """
        Run a set of sweeps on the network analyzer.
        Note that this will run all traces on the current channel.
        """
        root_instr = self.root_instrument
        # Store previous mode
        prev_mode = root_instr.sweep_mode()
        # Take instrument out of continuous mode, and send triggers equal to
        # the number of averages
        if root_instr.averages_enabled():
            avg = root_instr.averages()
            root_instr.reset_averages()
            root_instr.group_trigger_count(avg)
            root_instr.sweep_mode('GRO')
        else:
            root_instr.sweep_mode('SING')

        # Once the sweep mode is in hold, we know we're done
        try:
            while root_instr.sweep_mode() != 'HOLD':
                time.sleep(0.1)
        except KeyboardInterrupt:
            # If the user aborts because (s)he is stuck in the infinite loop
            # mentioned above, provide a hint of what can be wrong.
            msg = "User abort detected. "
            source = root_instr.trigger_source()
            if source == "MAN":
                msg += "The trigger source is manual. Are you sure this is " \
                       "correct? Please set the correct source with the " \
                       "'trigger_source' parameter"
            elif source == "EXT":
                msg += "The trigger source is external. Is the trigger " \
                       "source functional?"
            self.log.warning(msg)

        # Return previous mode, incase we want to restore this
        return prev_mode

    def write(self, cmd: str) -> None:
        """
        Select correct trace before querying
        """
        self.root_instrument.active_trace(self.trace_num)
        super().write(cmd)

    def ask(self, cmd: str) -> str:
        """
        Select correct trace before querying
        """
        self.root_instrument.active_trace(self.trace_num)
        return super().ask(cmd)

    def _Sparam(self) -> str:
        """
        Extrace S_parameter from returned PNA format
        """
        paramspec = self.root_instrument.get_trace_catalog()
        specs = paramspec.split(',')
        for spec_ind in range(len(specs)//2):
            name, param = specs[spec_ind*2:(spec_ind+1)*2]
            if name == self.trace_name:
                return param
        raise RuntimeError("Can't find selected trace on the PNA")

    def _set_Sparam(self, val: str) -> None:
        """
        Set an S-parameter, in the format S<a><b>, where a and b
        can range from 1-4
        """
        if not re.match("S[1-4][1-4]", val):
            raise ValueError("Invalid S parameter spec")
        self.write(f"CALC:PAR:MOD:EXT \"{val}\"")


class PNABase(VisaInstrument):
    """
    Base qcodes driver for Agilent/Keysight series PNAs
    http://na.support.keysight.com/pna/help/latest/Programming/GP-IB_Command_Finder/SCPI_Command_Tree.htm

    Note: Currently this driver only expects a single channel on the PNA. We
          can handle multiple traces, but using traces across multiple channels
          may have unexpected results.
    """

    def __init__(self,
                 name: str,
                 address: str,
                 # Set frequency ranges
                 min_freq: Union[int, float], max_freq: Union[int, float],
                 # Set power ranges
                 min_power: Union[int, float], max_power: Union[int, float],
                 nports: int, # Number of ports on the PNA
                 **kwargs: Any) -> None:
        super().__init__(name, address, terminator='\n', **kwargs)
        self.min_freq = min_freq
        self.max_freq = max_freq

        self.log.info("Initializing %s with power range %r-%r, freq range %r-%r.",
                      name, min_power, max_power, min_freq, max_freq)

        #Ports
        ports = ChannelList(self, "PNAPorts", PNAPort)
        for port_num in range(1, nports+1):
            port = PNAPort(self, f"port{port_num}", port_num,
                           min_power, max_power)
            ports.append(port)
            self.add_submodule(f"port{port_num}", port)
        ports.lock()
        self.add_submodule("ports", ports)

        # Drive power
        self.add_parameter('power',
                           label='Power',
                           get_cmd='SOUR:POW?',
                           get_parser=float,
                           set_cmd='SOUR:POW {:.2f}',
                           unit='dBm',
                           vals=Numbers(min_value=min_power,
                                        max_value=max_power))

        # IF bandwidth
        self.add_parameter('if_bandwidth',
                           label='IF Bandwidth',
                           get_cmd='SENS:BAND?',
                           get_parser=float,
                           set_cmd='SENS:BAND {:.2f}',
                           unit='Hz',
                           vals=Numbers(min_value=1, max_value=15e6))

        # Number of averages (also resets averages)
        self.add_parameter('averages_enabled',
                           label='Averages Enabled',
                           get_cmd="SENS:AVER?",
                           set_cmd="SENS:AVER {}",
                           val_mapping={True: '1', False: '0'})
        self.add_parameter('averages',
                           label='Averages',
                           get_cmd='SENS:AVER:COUN?',
                           get_parser=int,
                           set_cmd='SENS:AVER:COUN {:d}',
                           unit='',
                           vals=Numbers(min_value=1, max_value=65536))

        # Setting frequency range
        self.add_parameter('start',
                           label='Start Frequency',
                           get_cmd='SENS:FREQ:STAR?',
                           get_parser=float,
                           set_cmd='SENS:FREQ:STAR {}',
                           unit='Hz',
                           vals=Numbers(min_value=min_freq,
                                        max_value=max_freq))
        self.add_parameter('stop',
                           label='Stop Frequency',
                           get_cmd='SENS:FREQ:STOP?',
                           get_parser=float,
                           set_cmd='SENS:FREQ:STOP {}',
                           unit='Hz',
                           vals=Numbers(min_value=min_freq,
                                        max_value=max_freq))
        self.add_parameter('center',
                           label='Center Frequency',
                           get_cmd='SENS:FREQ:CENT?',
                           get_parser=float,
                           set_cmd='SENS:FREQ:CENT {}',
                           unit='Hz',
                           vals=Numbers(min_value=min_freq,
                                        max_value=max_freq))
        self.add_parameter('span',
                           label='Frequency Span',
                           get_cmd='SENS:FREQ:SPAN?',
                           get_parser=float,
                           set_cmd='SENS:FREQ:SPAN {}',
                           unit='Hz',
                           vals=Numbers(min_value=min_freq,
                                        max_value=max_freq))
        self.add_parameter('cw',
                           label='CW Frequency',
                           get_cmd='SENS:FREQ:CW?',
                           get_parser=float,
                           set_cmd='SENS:FREQ:CW {}',
                           unit='Hz',
                           vals=Numbers(min_value=min_freq,
                                        max_value=max_freq))

        # Number of points in a sweep
        self.add_parameter('points',
                           label='Points',
                           get_cmd='SENS:SWE:POIN?',
                           get_parser=int,
                           set_cmd='SENS:SWE:POIN {}',
                           unit='',
                           vals=Numbers(min_value=1, max_value=100001))

        # Electrical delay
        self.add_parameter('electrical_delay',
                           label='Electrical Delay',
                           get_cmd='CALC:CORR:EDEL:TIME?',
                           get_parser=float,
                           set_cmd='CALC:CORR:EDEL:TIME {:.6e}',
                           unit='s',
                           vals=Numbers(min_value=0, max_value=100000))

        # Sweep Time
        self.add_parameter('sweep_time',
                           label='Time',
                           get_cmd='SENS:SWE:TIME?',
                           get_parser=float,
                           unit='s',
                           vals=Numbers(0, 1e6))
        # Sweep Mode
        self.add_parameter('sweep_mode',
                           label='Mode',
                           get_cmd='SENS:SWE:MODE?',
                           set_cmd='SENS:SWE:MODE {}',
                           vals=Enum("HOLD", "CONT", "GRO", "SING"))
        # Sweep Type
        self.add_parameter('sweep_type',
                           label='Type',
                           get_cmd='SENS:SWE:TYPE?',
                           set_cmd='SENS:SWE:TYPE {}',
                           vals=Enum('LIN', 'LOG', 'POW', 'CW', 'SEGM', 'PHAS'))

        # Group trigger count
        self.add_parameter('group_trigger_count',
                           get_cmd="SENS:SWE:GRO:COUN?",
                           get_parser=int,
                           set_cmd="SENS:SWE:GRO:COUN {}",
                           vals=Ints(1, 2000000))
        # Trigger Source
        self.add_parameter('trigger_source',
                           get_cmd="TRIG:SOUR?",
                           set_cmd="TRIG:SOUR {}",
                           vals=Enum("EXT", "IMM", "MAN"))

        # Axis Parameters
        self.add_parameter('frequency_axis',
                           unit='Hz',
                           label="Frequency",
                           parameter_class=PNAAxisParameter,
                           startparam=self.start,
                           stopparam=self.stop,
                           pointsparam=self.points,
                           vals=Arrays(shape=(self.points,)))
        self.add_parameter('frequency_log_axis',
                           unit='Hz',
                           label="Frequency",
                           parameter_class=PNALogAxisParamter,
                           startparam=self.start,
                           stopparam=self.stop,
                           pointsparam=self.points,
                           vals=Arrays(shape=(self.points,)))
        self.add_parameter('time_axis',
                           unit='s',
                           label="Time",
                           parameter_class=PNATimeAxisParameter,
                           startparam=None,
                           stopparam=self.sweep_time,
                           pointsparam=self.points,
                           vals=Arrays(shape=(self.points,)))

        # Traces
        self.add_parameter('active_trace',
                           label='Active Trace',
                           get_cmd="CALC:PAR:MNUM?",
                           get_parser=int,
                           set_cmd="CALC:PAR:MNUM {}",
                           vals=Numbers(min_value=1, max_value=24))
        # Note: Traces will be accessed through the traces property which
        # updates the channellist to include only active trace numbers
        self._traces = ChannelList(self, "PNATraces", PNATrace)
        self.add_submodule("traces", self._traces)
        # Add shortcuts to first trace
        trace1 = self.traces[0]
        params = trace1.parameters
        if not isinstance(params, dict):
            raise RuntimeError(f"Expected trace.parameters to be a dict got "
                               f"{type(params)}")
        for param in params.values():
            self.parameters[param.name] = param
        # And also add a link to run sweep
        self.run_sweep = trace1.run_sweep
        # Set this trace to be the default (it's possible to end up in a
        # situation where no traces are selected, causing parameter snapshots
        # to fail)
        self.active_trace(trace1.trace_num)

        # Set auto_sweep parameter
        # If we want to return multiple traces per setpoint without sweeping
        # multiple times, we should set this to false
        self.add_parameter('auto_sweep',
                           label='Auto Sweep',
                           set_cmd=None,
                           get_cmd=None,
                           vals=Bool(),
                           initial_value=True)

        # A default output format on initialisation
        self.write('FORM REAL,32')
        self.write('FORM:BORD NORM')

        self.connect_message()

    @property
    def traces(self) -> ChannelList:
        """
        Update channel list with active traces and return the new list
        """
        # Keep track of which trace was active before. This command may fail
        # if no traces were selected.
        try:
            active_trace = self.active_trace()
        except errors.VisaIOError as e:
            self.log.debug("Exception on querying active trace: %r", e)
            if e.error_code == errors.StatusCode.error_timeout:
                self.log.info("No active trace on PNA")
                active_trace = None
            else:
                raise

        # Get a list of traces from the instrument and fill in the traces list
        parlist = self.get_trace_catalog().split(",")
        self._traces.clear()
        for trace_name in parlist[::2]:
            trace_num = self.select_trace_by_name(trace_name)
            pna_trace = PNATrace(self, f"tr{trace_num}",
                                 trace_name, trace_num)
            self._traces.append(pna_trace)

        # Restore the active trace if there was one
        if active_trace:
            self.active_trace(active_trace)

        # Return the list of traces on the instrument
        return self._traces

    def get_options(self) -> Sequence[str]:
        # Query the instrument for what options are installed
        return self.ask('*OPT?').strip('"').split(',')

    def get_trace_catalog(self) -> str:
        """
        Get the trace catalog, that is a list of trace and sweep types
        from the PNA.

        The format of the returned trace is:
            trace_name,trace_type,trace_name,trace_type...
        """
        return self.ask("CALC:PAR:CAT:EXT?").strip('"')

    def select_trace_by_name(self, trace_name: str) -> int:
        """
        Select a trace on the PNA by name.

        Returns:
            The trace number of the selected trace
        """
        self.write(f"CALC:PAR:SEL '{trace_name}'")
        return self.active_trace()

    def reset_averages(self) -> None:
        """
        Reset averaging
        """
        self.write("SENS:AVER:CLE")

    def averages_on(self) -> None:
        """
        Turn on trace averaging
        """
        self.averages_enabled(True)

    def averages_off(self) -> None:
        """
        Turn off trace averaging
        """
        self.averages_enabled(False)

    def _set_power_limits(self,
                          min_power: Union[int, float],
                          max_power: Union[int, float]) -> None:
        """
        Set port power limits
        """
        self.power.vals = Numbers(min_value=min_power,
                                  max_value=max_power)
        for port in self.ports:
            port._set_power_limits(min_power, max_power)


class PNAxBase(PNABase):
    def _enable_fom(self) -> None:
        '''
        PNA-x units with two sources have an enormous list of functions &
        configurations. In practice, most of this will be set up manually on
        the unit, with power and frequency varied in a sweep.
        '''
        self.add_parameter('aux_frequency',
                           label='Aux Frequency',
                           get_cmd='SENS:FOM:RANG4:FREQ:CW?',
                           get_parser=float,
                           set_cmd='SENS:FOM:RANG4:FREQ:CW {:.2f}',
                           unit='Hz',
                           vals=Numbers(min_value=self.min_freq,
                                        max_value=self.max_freq))
