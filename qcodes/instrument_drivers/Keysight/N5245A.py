from cmath import phase
import math
import numpy as np
from qcodes import VisaInstrument, MultiParameter
from qcodes.utils.validators import Numbers
import time


def _Sparam(string):
    """
    Converting S-parameter of active trace to a number for qcodes loop
    """
    x = string.strip('"')
    idx = x.find('_S')
    return int(x[idx+2:idx+4])

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
        

class N5245A(VisaInstrument):
    """
    qcodes driver for Agilent N5245A Network Analyzer. Command list at
    http://na.support.keysight.com/pna/help/latest/Programming/GP-IB_Command_Finder/SCPI_Command_Tree.htm
    """

    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, terminator='\n', **kwargs)
        
        # A default measurement and output format on initialisation
        self.write('FORM ASCii,0')
        self.write('FORM:BORD SWAP')
        self.write('CALC:PAR:DEF:EXT "S21", S21')
        self.write('CALC:FORM MLOG')

        # Query the instrument for what options are installed
        options = self.ask('*OPT?').strip('"').split(',')
        
        '''
        Parameters
        '''
        # The active trace (i.e. what is pulled from the PNA), and its assigned S parameter.
        # To extract multiple S-parameters, set the trace first, then use one of the sweeps.
        self.add_parameter('trace_number',
                           label='Trace No',
                           get_cmd='CALC:PAR:MNUM?',
                           set_cmd='CALC:PAR:MNUM {}',
                           get_parser=int
                           )
        self.add_parameter('trace',
                           label='Trace',
                           get_cmd='CALC:PAR:SEL?',
                           get_parser=_Sparam
                           )
        
        # Drive power
        min_power = -90 if '419' in options or '219' in options else -30
        self.add_parameter('power',
                           label='Power',
                           get_cmd='SOUR:POW?',
                           get_parser=float,
                           set_cmd='SOUR:POW {:.2f}',
                           unit='dBm',
                           vals=Numbers(min_value=min_power,max_value=10))
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
        PNA-x units with two sources have an enormous list of functions & configurations.
        In practice, most of this will be set up manually on the unit, with power and frequency
        varied.
        '''
        if '400' in options or '224' in options:
            ports = ['1','2','3','4'] if '400' in options else ['1','2']
            for p in ports:
                power_cmd = 'SOUR:POW{}'.format(p)
                self.add_parameter('aux_power{}'.format(p),
                                   label='Aux Power',
                                   get_cmd=power_cmd + '?',
                                   get_parser=float,
                                   set_cmd=power_cmd+ ' {:.2f}',
                                   unit='dBm',
                                   vals=Numbers(min_value=min_power, max_value=10))
            self.add_parameter('aux_frequency',
                               label='Aux Frequency',
                               get_cmd='SENS:FOM:RANG4:FREQ:CW?',
                               get_parser=float,
                               set_cmd='SENS:FOM:RANG4:FREQ:CW {:.2f}',
                               unit='Hz',
                               vals=Numbers(min_value=10e6,max_value=50e9))
        
        '''
        Sweeps
        '''
        # An immediate collection of data from the instrument
        self.add_parameter('raw_trace',
                           start=self.start(),
                           stop=self.stop(),
                           npts=self.points(),
                           parameter_class=RawSweep)
        
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
                
        self.connect_message()


    def get_idn(self):
        IDN = self.ask_raw('*IDN?')
        vendor, model, serial, firmware = map(str.strip, IDN.split(','))
        IDN = {'vendor': vendor, 'model': model,
               'serial': serial, 'firmware': firmware}
        return IDN
        
    # Adjusting the sweep requires an adjustment to the sweep parameters
    def _set_start(self, val):
        self.write('SENS:FREQ:STAR {:.2f}'.format(val))
        self.raw_trace.set_sweep(val, self.stop(), self.points())
        self.vector_trace.set_sweep(val, self.stop(), self.points())
            
    def _set_stop(self, val):
        self.write('SENS:FREQ:STOP {:.2f}'.format(val))
        self.raw_trace.set_sweep(self.start(), val, self.points())
        self.vector_trace.set_sweep(self.start(), val, self.points())
            
    def _set_points(self, val):
        self.write('SENS:SWE:POIN {:d}'.format(val))
        self.raw_trace.set_sweep(self.start(), self.stop(), val)
        self.vector_trace.set_sweep(self.start(), self.stop(), val)


        
                
