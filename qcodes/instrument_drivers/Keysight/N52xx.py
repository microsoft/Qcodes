from cmath import phase
import math
import numpy as np
from qcodes import VisaInstrument, MultiParameter, InstrumentChannel
from qcodes.utils.validators import Numbers
import time
import re

class RawSweep(MultiParameter):
    """
    RawSweep pulls the magnitude and phase directly from the instrument without waiting.
    """
    def __init__(self, name, instrument, start, stop, npts):
        super().__init__(name, names=("",""), shapes=((), ()))
        self._instrument = instrument
        self.set_sweep(start, stop, npts)
        self.names = ('magnitude', 'phase')
        self.units = ('dB', 'rad')
        self.setpoint_names = (('frequency',), ('frequency',))
        
    def set_sweep(self, start, stop, npts):
        f = tuple(np.linspace(int(start), int(stop), num=npts))
        self.setpoints = ((f,), (f,))
        self.shapes = ((npts,), (npts,))
        
    def get(self):
        self.set_sweep(self._instrument.start(), self._instrument.stop(), self._instrument.points())
        
        data_str = self._instrument.ask('CALC:DATA? SDATA').split(',')
        data_list = [float(v) for v in data_str]

        # Unpack and convert to log magnitude and phase
        data_arr = np.array(data_list).reshape(int(len(data_list) / 2), 2)
        mag_array, phase_array = [], []
        for comp in data_arr:
            complex_num = complex(comp[0], comp[1])
            mag_array.append(20 * math.log10(abs(complex_num)))
            phase_array.append(phase(complex_num))
        return mag_array, phase_array
        
        
class MagPhaseSweep(MultiParameter):
    """
    MagPhase will run a sweep, including averaging, before returning data.
    As such, wait time in a loop is not needed.
    """
    def __init__(self, name, instrument, start, stop, npts):
        super().__init__(name, names=("",""), shapes=((), ()))
        self._instrument = instrument
        self.set_sweep(start, stop, npts)
        self.names = ('magnitude', 'phase')
        self.units = ('dB', 'degrees')
        self.setpoint_names = (('frequency',), ('frequency',))
        
    def set_sweep(self, start, stop, npts):
        f = tuple(np.linspace(int(start), int(stop), num=npts))
        self.setpoints = ((f,), (f,))
        self.shapes = ((npts,), (npts,))
        
    def get(self):
        self.set_sweep(self._instrument.start(), self._instrument.stop(), self._instrument.points())
        
        # Take instrument out of continuous mode, and send triggers equal to the number of averages
        self._instrument.write('SENS:AVER:CLE')
        self._instrument.write('TRIG:SOUR IMM')
        avg = self._instrument.averages()
        self._instrument.write('SENS:SWE:GRO:COUN {0}'.format(avg))
        self._instrument.write('SENS:SWE:MODE GRO')

        # Once the sweep mode is in hold, we know we're done
        while True:
            if self._instrument.ask('SENS:SWE:MODE?') == 'HOLD':
                break
            time.sleep(0.2)
        
        # Ask for magnitude
        previous_format = self._instrument.ask('CALC:FORM?')
        self._instrument.write('CALC:FORM MLOG')
        mag_str = self._instrument.ask('CALC:DATA? FDATA').split(',')
        mag_list = [float(v) for v in mag_str]
        
        # Then phase
        self._instrument.write('CALC:FORM UPH')
        phase_str = self._instrument.ask('CALC:DATA? FDATA').split(',')
        phase_list = [float(v) for v in phase_str]

        # Return the instrument state
        self._instrument.write('CALC:FORM {}'.format(previous_format))
        self._instrument.write('SENS:SWE:MODE CONT')
                      
        return mag_list, phase_list

class FormattedSweep(ArrayParameter):
    """
    Mag will run a sweep, including averaging, before returning data.
    As such, wait time in a loop is not needed.
    """
    def __init__(self, name, instrument, format, label, unit, start, stop, npts):
        super().__init__(name,
                         label=label,
                         unit=unit,
                         shape=(npts,),
                         setpoints=(np.linspace(start, stop, num=npts),),
                         setpoint_names=('frequency',),
                         setpoint_labels=('Frequency',),
                         setpoint_units=('Hz',)
                         )
        self._instrument = instrument
        self.format = format
        
    def get_raw(self):
        
        # Take instrument out of continuous mode, and send triggers equal to the number of averages
        self._instrument.write('SENS:AVER:CLE')
        self._instrument.write('TRIG:SOUR IMM')
        avg = self._instrument.averages()
        self._instrument.write('SENS:SWE:GRO:COUN {0}'.format(avg))
        self._instrument.write('SENS:SWE:MODE GRO')

        # Once the sweep mode is in hold, we know we're done
        while True:
            if self._instrument.ask('SENS:SWE:MODE?') == 'HOLD':
                break
            time.sleep(0.1)
        
        # Ask for magnitude
        self._instrument.write('CALC:FORM %s' % self.format)
        data_str = self._instrument.ask('CALC:DATA? FDATA').split(',')
        data_list = np.fromiter((float(v) for v in data_str), float)

        # Return the instrument state
        self._instrument.write('SENS:SWE:MODE CONT')
                      
        return data_list

    @property
    def shape(self):
        return (self._instrument.root_instrument.points(),)
    @property
    def setpoints(self):
        start = self._instrument.root_instrument.start()
        stop = self._instrument.root_instrument.stop()
        return (np.linspace(start, stop, self.shape),)
    

class PNAPort(InstrumentChannel):
    """
    Allow operations on individual PNA ports.
    Note: This can be expanded to include a large number of extra parameters...
    """

    def __init__(self, parent, name, port,
                 min_power, max_power):
        super().__init__(parent, name)

        self.port = int(port)
        if self.port < 1 or self.port > 4:
            raise ValueError("Port must be between 1 and 4.")

        pow_cmd = f"SOUR:POW{self.port}"
        self.add_parameter("source_power",
                           label="power",
                           unit="dBm",
                           get_cmd=pow_cmd + "?",
                           set_cmd=pow_cmd + "{}",
                           get_parser=float)

class PNATrace(InstrumentChannel):
    """
    Allow operations on individual PNA traces.
    """

    def __init__(self, parent, name, trace):
        super().__init__(parent, name)
        self.trace = trace

        # Name of parameter
        self.add_parameter('trace',
                           label='Trace',
                           get_cmd='CALC:PAR:SEL?'.format(self.trace),
                           get_parser=_Sparam
                           )

        # And a list of individual formats
        self.add_parameter('magnitude',
                           format='MLOG',
                           label='Magnitude',
                           unit='dB',
                           start=start,
                           stop=stop,
                           npts=npts,
                           parameter_class=FormattedSweep)
        self.add_parameter('phase',
                           format='PHAS',
                           label='Phase',
                           unit='deg',
                           start=start,
                           stop=stop,
                           npts=npts,
                           parameter_class=FormattedSweep)
        self.add_parameter('real',
                           format='REAL',
                           label='Real',
                           unit='LinMag',
                           start=start,
                           stop=stop,
                           npts=npts,
                           parameter_class=FormattedSweep)
        self.add_parameter('imaginary',
                           format='IMAG',
                           label='Imaginary',
                           unit='LinMag',
                           start=start,
                           stop=stop,
                           npts=npts,
                           parameter_class=FormattedSweep)

    def write(self, cmd):
        """
        Select correct trace before querying
        """
        super().write("CALC:PAR:MNUM {}".format(self.trace))
        super().write(cmd)
    def ask(self, cmd):
        """
        Select correct trace before querying
        """
        super().write("CALC:PAR:MNUM {}".format(self.trace))
        return super().ask(cmd)

    @staticmethod
    def parse_paramstring(paramspec):
        """
        Parse parameter specification from PNA
        """
        paramspec = paramspec.strip('"')
        ch, param, trnum = re.findall(r"CH(\d+)_(S\d+)_(\d+)", paramspec)[0]
        return ch, param, trnum

    def _Sparam(self, paramspec):
        """
        Extrace S_parameter from returned PNA format
        """
        return self.parse_paramstring(paramspec)[1]

class PNABase(VisaInstrument):
    """
    Base qcodes driver for Agilent/Keysight series PNAs
    http://na.support.keysight.com/pna/help/latest/Programming/GP-IB_Command_Finder/SCPI_Command_Tree.htm

    Note: Currently this driver only expects a single channel on the PNA. We can handle multiple
          traces, but using traces across multiple channels may have unexpected results.
    """

    def __init__(self, name, address, 
                 min_freq, max_freq, # Set frequency ranges
                 min_power, max_power, # Set power ranges
                 nports, # Number of ports on the PNA
                 **kwargs):
        super().__init__(name, address, terminator='\n', **kwargs)

        #Ports
        ports = ChannelList(self, "PNAPorts", PNAPort)
        for port in range(1,nports+1):
            port = PNAPort(self, f"Port{port}", port)
            ports.append(port)
            self.add_submodule(port)
        ports.lock()
        self.add_submodule("ports", ports)

        # Traces
        # Note: These will be accessed through the traces property which updates
        # the channellist to include only active trace numbers
        self._traces = ChannelList(self, "PNATraces", PNATrace)
        self.add_submodule("traces", self._traces)
        
        # Parameters
        # The active trace (i.e. what is pulled from the PNA), and its assigned S parameter.
        # To extract multiple S-parameters, set the trace first, then use one of the sweeps.
        self.add_parameter('trace_number',
                           label='Trace No',
                           get_cmd='CALC:PAR:MNUM?',
                           set_cmd='CALC:PAR:MNUM {}',
                           get_parser=int
                           )

        # Drive power
        self.add_parameter('power',
                           label='Power',
                           get_cmd='SOUR:POW?',
                           get_parser=float,
                           set_cmd='SOUR:POW {:.2f}',
                           unit='dBm',
                           vals=Numbers(min_value=min_power,max_value=max_power))
        self.add_parameter('sweep_time',
                           label='Time',
                           get_cmd='SENS:SWE:TIME?',
                           get_parser=float,
                           unit='s',
                           vals=Numbers(0,1e6))
        
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
                            val_mapping={True: 'ON', False: 'OFF'})
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
                           set_cmd=self._set_start,
                           unit='',
                           vals=Numbers(min_value=10e6,max_value=50.2e9))
        self.add_parameter('stop',
                           label='Stop Frequency',
                           get_cmd='SENS:FREQ:STOP?',
                           get_parser=float,
                           set_cmd=self._set_stop,
                           unit='',
                           vals=Numbers(min_value=10e6,max_value=50.2e9))
        
        # Number of points in a sweep
        self.add_parameter('points',
                           label='Points',
                           get_cmd='SENS:SWE:POIN?',
                           get_parser=int,
                           set_cmd=self._set_points,
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
        
        '''
        Sweeps
        '''
        # Wait until the sweep (and averages) are completed first
        self.add_parameter('vector_trace',
                           start = self.start(),
                           stop = self.stop(),
                           npts = self.points(),
                           parameter_class=MagPhaseSweep)
                    
        '''
        Functions
        '''
        # Clear averages
        self.add_function('reset_averages',
                          call_cmd='SENS:AVER:CLE')
        # Averages ON
        self.add_function('averages_on',
                          call_cmd='SENS:AVER ON')
        # Averages OFF
        self.add_function('averages_off',
                          call_cmd='SENS:AVER OFF')

        # A default output format on initialisation
        self.write('FORM ASCii,0')
        self.write('FORM:BORD SWAP')

        self.connect_message()

    @property
    def traces(self):
        """
        Update channel list with active traces and return the new list
        """
        parlist = self.ask("CALC:PAR:CAT:EXT?").strip('"').split(",")
        self._traces.clear()
        for trace in parlist[::2]:
            trnum = PNATrace.parse_paramstring(trace)[2]
            trace = PNATrace(self, "tr{}".format(trnum), int(trnum))
            self._traces.add(trace)
        return self._traces
    

    def get_options(self):
        # Query the instrument for what options are installed
        return self.ask('*OPT?').strip('"').split(',')

    def _set_power_limits(self, min_power, max_power):
        """
        Set port power limits
        """
        self.power.vals = Numbers(min_value=min_power,max_value=max_power)

    # Adjusting the sweep requires an adjustment to the sweep parameters
    def _set_start(self, val):
        self.write('SENS:FREQ:STAR {:.2f}'.format(val))
        self._set_sweep(start=val)
            
    def _set_stop(self, val):
        self.write('SENS:FREQ:STOP {:.2f}'.format(val))
        self._set_sweep(stop=val)
            
    def _set_points(self, val):
        self.write('SENS:SWE:POIN {:d}'.format(val))
        self._set_sweep(npts=val)

    def _set_sweep(self, start=None, stop=None, npts=None):
        if start is None:
            start = self.start()
        if stop is None:
            stop = self.stop()
        if npts is None:
            npts = self.points()
        for trace in self.traces:
            trace.set_sweep(start, stop, npts)

class PNAxBase(PNABase):
    def __init__(self, name, address, 
                 min_freq, max_freq, # Set frequency ranges
                 min_power, max_power, # Set power ranges
                 nports, # Number of ports on the PNA
                 **kwargs):

        super.__init__(self, name, address, 
                       min_freq, max_freq,
                       min_power, max_power,
                       **kwargs)

    def _enable_fom(self):
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