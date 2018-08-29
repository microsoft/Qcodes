import numpy as np
from qcodes import VisaInstrument, InstrumentChannel, ArrayParameter, ChannelList
from qcodes.utils.validators import Numbers, Enum, Bool
from typing import Sequence, Union, Any, Tuple
import time
import re

class PNASweep(ArrayParameter):
    def __init__(self,
                 name: str,
                 instrument: 'PNABase',
                 **kwargs: Any) -> None:

        super().__init__(name,
                 instrument=instrument,
                 shape=(0,),
                 setpoints=((0,),),
                 **kwargs
                 )

    @property # type: ignore
    def shape(self) -> Sequence[int]: # type: ignore
        if self._instrument is None:
            return (0,)
        return (self._instrument.root_instrument.points(),)
    @shape.setter
    def shape(self, val: Sequence[int]) -> None:
        pass

    @property # type: ignore
    def setpoints(self) -> Sequence: # type: ignore
        start = self._instrument.root_instrument.start()
        stop = self._instrument.root_instrument.stop()
        return (np.linspace(start, stop, self.shape[0]),)
    @setpoints.setter
    def setpoints(self, val: Sequence[int]) -> None:
        pass

class FormattedSweep(PNASweep):
    """
    Mag will run a sweep, including averaging, before returning data.
    As such, wait time in a loop is not needed.
    """
    def __init__(self,
                 name: str,
                 instrument: 'PNABase',
                 sweep_format: str,
                 label: str,
                 unit: str,
                 memory: bool=False) -> None:
        super().__init__(name,
                         instrument=instrument,
                         label=label,
                         unit=unit,
                         setpoint_names=('frequency',),
                         setpoint_labels=('Frequency',),
                         setpoint_units=('Hz',)
                         )
        self.sweep_format = sweep_format
        self.memory = memory

    def get_raw(self) -> Sequence[float]:
        root_instr = self._instrument.root_instrument
        # Check if we should run a new sweep
        if root_instr.auto_sweep():
            prev_mode = self._instrument.run_sweep()
        # Ask for data, setting the format to the requested form
        self._instrument.write(f'CALC:FORM {self.sweep_format}')
        data = np.array(root_instr.visa_handle.query_binary_values('CALC:DATA? FDATA', datatype='f', is_big_endian=True))
        # Restore previous state if it was changed
        if root_instr.auto_sweep():
            root_instr.sweep_mode(prev_mode)

        return data

class PNAPort(InstrumentChannel):
    """
    Allow operations on individual PNA ports.
    Note: This can be expanded to include a large number of extra parameters...
    """

    def __init__(self,
                 parent: 'PNABase',
                 name: str,
                 port: int,
                 min_power: Union[int, float],
                 max_power: Union[int, float]) -> None:
        super().__init__(parent, name)

        self.port = int(port)
        if self.port < 1 or self.port > 4:
            raise ValueError("Port must be between 1 and 4.")

        pow_cmd = f"SOUR:POW{self.port}"
        self.add_parameter("source_power",
                           label="power",
                           unit="dBm",
                           get_cmd=f"{pow_cmd}?",
                           set_cmd=f"{pow_cmd} {{}}",
                           get_parser=float)

    def _set_power_limits(self,
                          min_power: Union[int, float],
                          max_power: Union[int, float]) -> None:
        """
        Set port power limits
        """
        self.source_power.vals = Numbers(min_value=min_power,max_value=max_power)

class PNATrace(InstrumentChannel):
    """
    Allow operations on individual PNA traces.
    """

    def __init__(self,
                 parent: 'PNABase',
                 name: str,
                 trace: int) -> None:
        super().__init__(parent, name)
        self.trace = trace

        # Name of parameter (i.e. S11, S21 ...)
        self.add_parameter('trace',
                           label='Trace',
                           get_cmd='CALC:PAR:SEL?',
                           get_parser=self._Sparam,
                           set_cmd=self._set_Sparam
                           )
        # Format
        # Note: Currently parameters that return complex values are not supported
        # as there isn't really a good way of saving them into the dataset
        self.add_parameter('format',
                           label='Format',
                           get_cmd='CALC:FORM?',
                           set_cmd='CALC:FORM {}',
                           vals=Enum('MLIN', 'MLOG', 'PHAS', 'UPH', 'IMAG', 'REAL'))

        # And a list of individual formats
        self.add_parameter('magnitude',
                           sweep_format='MLOG',
                           label='Magnitude',
                           unit='dB',
                           parameter_class=FormattedSweep)
        self.add_parameter('linear_magnitude',
                           sweep_format='MLIN',
                           label='Magnitude',
                           unit='ratio',
                           parameter_class=FormattedSweep)
        self.add_parameter('phase',
                           sweep_format='PHAS',
                           label='Phase',
                           unit='deg',
                           parameter_class=FormattedSweep)
        self.add_parameter('unwrapped_phase',
                           sweep_format='UPH',
                           label='Phase',
                           unit='deg',
                           parameter_class=FormattedSweep)
        self.add_parameter("group_delay",
                           sweep_format='GDEL',
                           label='Group Delay',
                           unit='s',
                           parameter_class=FormattedSweep)
        self.add_parameter('real',
                           sweep_format='REAL',
                           label='Real',
                           unit='LinMag',
                           parameter_class=FormattedSweep)
        self.add_parameter('imaginary',
                           sweep_format='IMAG',
                           label='Imaginary',
                           unit='LinMag',
                           parameter_class=FormattedSweep)

    def run_sweep(self) -> str:
        root_instr = self.root_instrument
        # Store previous mode
        prev_mode = root_instr.sweep_mode()
        # Take instrument out of continuous mode, and send triggers equal to the number of averages
        if root_instr.averages_enabled:
            avg = root_instr.averages()
            root_instr.write('SENS:AVER:CLE')
            root_instr.write('SENS:SWE:GRO:COUN {0}'.format(avg))
            root_instr.root_instrument.sweep_mode('GRO')
        else:
            root_instr.root_instrument.sweep_mode('SING')

        # Once the sweep mode is in hold, we know we're done
        while root_instr.sweep_mode() != 'HOLD':
            time.sleep(0.1)

        # Return previous mode, incase we want to restore this
        return prev_mode

    def write(self, cmd: str) -> None:
        """
        Select correct trace before querying
        """
        self.root_instrument.active_trace(self.trace)
        super().write(cmd)

    def ask(self, cmd: str) -> str:
        """
        Select correct trace before querying
        """
        self.root_instrument.active_trace(self.trace)
        return super().ask(cmd)

    @staticmethod
    def parse_paramstring(paramspec: str) -> Tuple[str, str, str]:
        """
        Parse parameter specification from PNA
        """
        paramspec = paramspec.strip('"')
        ch, param, trnum = re.findall(r"CH(\d+)_(S\d+)_(\d+)", paramspec)[0]
        return ch, param, trnum

    def _Sparam(self, paramspec: str) -> str:
        """
        Extrace S_parameter from returned PNA format
        """
        return self.parse_paramstring(paramspec)[1]

    def _set_Sparam(self, val: str) -> None:
        """
        Set an S-parameter, in the format S<a><b>, where a and b
        can range from 1-4
        """
        if not re.match("S[1-4][1-4]", val):
            raise ValueError("Invalid S parameter spec")
        self.write(f"CALC:PAR:MOD:EXT {val}")

class PNABase(VisaInstrument):
    """
    Base qcodes driver for Agilent/Keysight series PNAs
    http://na.support.keysight.com/pna/help/latest/Programming/GP-IB_Command_Finder/SCPI_Command_Tree.htm

    Note: Currently this driver only expects a single channel on the PNA. We can handle multiple
          traces, but using traces across multiple channels may have unexpected results.
    """

    def __init__(self,
                 name: str,
                 address: str,
                 min_freq: Union[int, float], max_freq: Union[int, float], # Set frequency ranges
                 min_power: Union[int, float], max_power: Union[int, float], # Set power ranges
                 nports: int, # Number of ports on the PNA
                 **kwargs: Any) -> None:
        super().__init__(name, address, terminator='\n', **kwargs)

        #Ports
        ports = ChannelList(self, "PNAPorts", PNAPort)
        for port_num in range(1,nports+1):
            port = PNAPort(self, f"port{port_num}", port_num, min_power, max_power)
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
                           vals=Numbers(min_value=min_power,max_value=max_power))

        # IF bandwidth
        self.add_parameter('if_bandwidth',
                           label='IF Bandwidth',
                           get_cmd='SENS:BAND?',
                           get_parser=float,
                           set_cmd='SENS:BAND {:.2f}',
                           unit='Hz',
                           vals=Numbers(min_value=1,max_value=15e6))

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
                           vals=Numbers(min_value=1,max_value=65536))

        # Setting frequency range
        self.add_parameter('start',
                           label='Start Frequency',
                           get_cmd='SENS:FREQ:STAR?',
                           get_parser=float,
                           set_cmd='SENS:FREQ:STAR {}',
                           unit='',
                           vals=Numbers(min_value=min_freq,max_value=max_freq))
        self.add_parameter('stop',
                           label='Stop Frequency',
                           get_cmd='SENS:FREQ:STOP?',
                           get_parser=float,
                           set_cmd='SENS:FREQ:STOP {}',
                           unit='',
                           vals=Numbers(min_value=min_freq,max_value=max_freq))

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
                           vals=Numbers(0,1e6))
        # Sweep Mode
        self.add_parameter('sweep_mode',
                           label='Mode',
                           get_cmd='SENS:SWE:MODE?',
                           set_cmd='SENS:SWE:MODE {}',
                           vals=Enum("HOLD", "CONT", "GRO", "SING"))

        # Traces
        self.add_parameter('active_trace',
                           label='Active Trace',
                           get_cmd="CALC:PAR:MNUM?",
                           get_parser=int,
                           set_cmd="CALC:PAR:MNUM {}",
                           vals=Numbers(min_value=1, max_value=24))
        # Note: Traces will be accessed through the traces property which updates
        # the channellist to include only active trace numbers
        self._traces = ChannelList(self, "PNATraces", PNATrace)
        self.add_submodule("traces", self._traces)
        # Add shortcuts to trace 1
        trace1 = PNATrace(self, "tr1", 1)
        for param in trace1.parameters.values():
            self.parameters[param.name] = param
        # Set this trace to be the default (it's possible to end up in a situation where
        # no traces are selected, causing parameter snapshots to fail)
        self.active_trace(1)

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
        parlist = self.ask("CALC:PAR:CAT:EXT?").strip('"').split(",")
        self._traces.clear()
        for trace in parlist[::2]:
            trnum = PNATrace.parse_paramstring(trace)[2]
            pna_trace = PNATrace(self, "tr{}".format(trnum), int(trnum))
            self._traces.append(pna_trace)
        return self._traces

    def get_options(self) -> Sequence[str]:
        # Query the instrument for what options are installed
        return self.ask('*OPT?').strip('"').split(',')

    def reset_averages(self):
        """
        Reset averaging
        """
        self.write("SENS:AVER:CLE")

    def averages_on(self):
        """
        Turn on trace averaging
        """
        self.averages_enabled(True)

    def averages_off(self):
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
        self.power.vals = Numbers(min_value=min_power,max_value=max_power)
        for port in self.ports:
            port._set_power_limits(min_power, max_power)

class PNAxBase(PNABase):
    def __init__(self,
                 name: str,
                 address: str,
                 min_freq: Union[int, float], max_freq: Union[int, float], # Set frequency ranges
                 min_power: Union[int, float], max_power: Union[int, float], # Set power ranges
                 nports: int, # Number of ports on the PNA
                 **kwargs: Any) -> None:

        super().__init__(name, address,
                       min_freq, max_freq,
                       min_power, max_power,
                       nports,
                       **kwargs)

    def _enable_fom(self) -> None:
        '''
        PNA-x units with two sources have an enormous list of functions & configurations.
        In practice, most of this will be set up manually on the unit, with power and frequency
        varied.
        '''
        self.add_parameter('aux_frequency',
                           label='Aux Frequency',
                           get_cmd='SENS:FOM:RANG4:FREQ:CW?',
                           get_parser=float,
                           set_cmd='SENS:FOM:RANG4:FREQ:CW {:.2f}',
                           unit='Hz',
                           vals=Numbers(min_value=10e6,max_value=50e9))
