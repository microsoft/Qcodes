from qcodes import VisaInstrument
from qcodes.utils import validators as vals
from cmath import phase
import numpy as np
from qcodes import Parameter

class FrequencySweep(Parameter):
  '''
    This is the composite parameter class for a frequency sweep done with the Rohde Schwarz RSZNB20 
    it allows for 'fast redout' where the instrument returns an list of transmission data in the form 
    of a complex numbers taken from a frequency sweep.
  '''
	def __init__(self, name, instrument, start, stop, npts):
		super().__init__(name)
		self._instrument = instrument
		self.set_sweep(start, stop, npts)
		self.names = ('magnitude', 'phase')
		self.units = ('dBm', 'rad')
		self.setpoint_names = (('frequency',), ('frequency',))
 
	def set_sweep(self, start, stop, npts):
		# function to update the config of the software parameter (wouldn't be needed if we had dynamic sizing)
    # f is a tuple because it needs to be hashable in parameter.py
		f = tuple(np.linspace(int(start), int(stop), num=npts))
		self.setpoints = ((f,), (f,))
		self.shapes = ((npts,), (npts,))
 
	def get(self):
    # function to get the trace data for preset start, stop, npts and avg
		self._instrument.write('SENS1:AVER:STAT ON')
        self._instrument.write('AVER:CLE')
		self._instrument.turn_off_cont_meas()
    # instrument averages over its last 'avg' number of sweeps so this is needed to ensure the return is the result
		for avgcount in range(self._instrument.avg()-1):
      self._instrument.write('INIT:IMM; *WAI')
			self._instrument.write('INIT:IMM; *WAI')
		self._instrument.update_display_once()
		data_list = [float(v) for v in self._instrument.ask('CALC:DATA? SDAT').split(',')]
		# complex numbers returned in a list [re1,im1,re2,im2...] so needs processing
    data_arr = data_arr = np.array(data_list).reshape(int(len(data_list)/2),2)
    mag_array=[]
		phase_array=[]
		for comp in data_arr:
			complex_num = complex(comp[0],comp[1])
			mag_array.append(abs(complex_num))
			phase_array.append(phase(complex_num))
		return mag_array, phase_array


class ZNB20(VisaInstrument):
    '''
      This is the qcodes driver for the Rohde & Schwarz ZNB20 virtual network analyser

        TODO:
        - Add all parameters that are in the manual
        - Add test suite
        - Error handling
        - check initialisation settings and test functions
    '''
    def __init__(self, name, address, **kwargs):
        super().__init__(name=name, address=address, **kwargs)

        self.add_parameter(name='power',
                           label='Power',
                           units='dBm',
                           get_cmd='SOUR:POW?',
                           set_cmd='SOUR:POW {:d}',
                           get_parser=int,
                           vals=vals.Numbers(-150, 25))

        self.add_parameter(name='bandwidth',
                           label='Bandwidth',
                           units='Hz', 
                           get_cmd='SENS:BAND?',
                           set_cmd='SENS:BAND {:d}',
                           get_parser=int,
                           vals=vals.Numbers(1,1e6))

        self.add_parameter(name='avg',
                           label='Averages',
                           units='',
                           get_cmd='AVER:COUN?',
                           set_cmd='AVER:COUN {:d}',
                           get_parser=int,
                           vals=vals.Numbers(1,5000))

        self.add_parameter(name='start', 
                           get_cmd='SENS:FREQ:START?',
                           set_cmd=self._set_start,
						   get_parser=int,
                           )

        self.add_parameter(name='stop', 
                           get_cmd='SENS:FREQ:STOP?',
                           set_cmd=self._set_stop,
						   get_parser=int,
                           )

        self.add_parameter(name='npts',
                           get_cmd='SENS:SWE:POIN?',
                           set_cmd=self._set_npts,
                           get_parser=int
                           )

        self.add_parameter(name = 'trace',
						   start=self.start(),
                           stop=self.stop(),
                           npts=self.npts(),
                           parameter_class=FrequencySweep)       

        self.add_function('reset', call_cmd='*RST')
        self.add_function('turn_on_tooltip', call_cmd='SYST:ERR:DISP ON')
        self.add_function('turn_off_tooltip', call_cmd='SYST:ERR:DISP OFF')
        self.add_function('turn_on_cont_meas', call_cmd='INIT:CONT:ALL ON')
        self.add_function('turn_off_cont_meas', call_cmd='INIT:CONT:ALL OFF')
        self.add_function('update_display_once', call_cmd='SYST:DISP:UPD ONCE')
        self.add_function('update_display_on', call_cmd='SYST:DISP:UPD ON')
        self.add_function('update_display_off', call_cmd='SYST:DISP:UPD OFF')
        self.add_function('rf_off', call_cmd='OUTP1 OFF')

        self.initialise()
        self.connect_message()

    def _set_start(self, val):
        self.write('SENS:FREQ:START {:.4f}'.format(val))
        self.trace.set_sweep(val, self.stop(), self.npts())

    def _set_stop(self, val):
        self.write('SENS:FREQ:STOP {:.4f}'.format(val))
        self.trace.set_sweep(self.start(), val, self.npts())
 
    def _set_npts(self, val):
        self.write('SENS:SWE:POIN {:.4f}'.format(val))
        self.trace.set_sweep(self.start(), self.stop(), val)

    def initialise(self):
        self.write('*RST')
        self.write('SENS1:SWE:TYPE LIN')
        self.write('SENS1:SWE:TIME:AUTO ON')
        self.write('TRIG1:SEQ:SOUR IMM')
        self.write('SENS1:AVER:STAT ON')
        self._set_start(1e6)
        self._set_stop(2e6)
        self._set_npts(10)
