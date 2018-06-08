from cmath import phase
import math
import numpy as np
from qcodes import VisaInstrument, MultiParameter, InstrumentChannel, ArrayParameter, ChannelList
from qcodes.utils.validators import Numbers
import time
import re

class PNASweep(ArrayParameter):
    def __init__(self, name, instrument, **kwargs):
        super().__init__(name,
                 instrument=instrument,
                 shape=(0,),
                 setpoints=((0,),),
                 **kwargs
                 )

    @property
    def shape(self):
        if self._instrument is None:
            return (0,)
        return (self._instrument.root_instrument.points(),)
    @shape.setter
    def shape(self, val):
        pass
    @property
    def setpoints(self):
        start = self._instrument.root_instrument.start()
        stop = self._instrument.root_instrument.stop()
        return (np.linspace(start, stop, self.shape[0]),)
    @setpoints.setter
    def setpoints(self, val):
        pass

class FormattedSweep(PNASweep):
    """
    Mag will run a sweep, including averaging, before returning data.
    As such, wait time in a loop is not needed.
    """
    def __init__(self, name, instrument, format, label, unit):
        super().__init__(name,
                         instrument=instrument,
                         label=label,
                         unit=unit,
                         setpoint_names=('frequency',),
                         setpoint_labels=('Frequency',),
                         setpoint_units=('Hz',)
                         )
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
    
    def _set_power_limits(self, min_power, max_power):
        """
        Set port power limits
        """
        self.source_power.vals = Numbers(min_value=min_power,max_value=max_power)

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
                           get_parser=self._Sparam
                           )

        # And a list of individual formats
        self.add_parameter('magnitude',
                           format='MLOG',
                           label='Magnitude',
                           unit='dB',
                           parameter_class=FormattedSweep)
        self.add_parameter('phase',
                           format='PHAS',
                           label='Phase',
                           unit='deg',
                           parameter_class=FormattedSweep)
        self.add_parameter('real',
                           format='REAL',
                           label='Real',
                           unit='LinMag',
                           parameter_class=FormattedSweep)
        self.add_parameter('imaginary',
                           format='IMAG',
                           label='Imaginary',
                           unit='LinMag',
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
                           set_cmd='SENS:FREQ:STOP {}',
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

        # Traces
        # Note: These will be accessed through the traces property which updates
        # the channellist to include only active trace numbers
        self._traces = ChannelList(self, "PNATraces", PNATrace)
        self.add_submodule("traces", self._traces)
        # Add shortcuts to trace 1
        trace1 = PNATrace(self, "tr1", 1)
        for param in trace1.parameters.values():
            self.parameters[param.name] = param
        # By default we should also pull any following values from this trace
        self.write("CALC:PAR:MNUM 1")
                    
        # Functions
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
        self.write('FORM REAL,32')
        self.write('FORM:BORD NORM')

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
            self._traces.append(trace)
        return self._traces
    

    def get_options(self):
        # Query the instrument for what options are installed
        return self.ask('*OPT?').strip('"').split(',')

    def _set_power_limits(self, min_power, max_power):
        """
        Set port power limits
        """
        self.power.vals = Numbers(min_value=min_power,max_value=max_power)
        for port in self.ports:
            port._set_power_limits(min_power, max_power)

class PNAxBase(PNABase):
    def __init__(self, name, address, 
                 min_freq, max_freq, # Set frequency ranges
                 min_power, max_power, # Set power ranges
                 nports, # Number of ports on the PNA
                 **kwargs):

        super().__init__(name, address, 
                       min_freq, max_freq,
                       min_power, max_power,
                       nports,
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