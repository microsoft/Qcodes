from qcodes import VisaInstrument
from qcodes.utils import validators as vals
from cmath import phase
import numpy as np
from qcodes import Parameter

'''
    TODO: 
        better error messages (eg do I have enough to do a sweep?)
        add functionality (ability to set start and stop freq as well as centre and span)
        needs on/off status?
'''

class FrequencySweep(Parameter):
	def __init__(self, name, instrument, start, stop, npts, **kwargs):
		super().__init__(name)
		self.p = 'ex'
		self.name = name
		self._instrument = instrument
		self.set_sweep(start, stop, npts)
		self.names = ('magnitude', 'phase')
		self.units = ('dBm', 'degrees')
		self.setpoint_names = (('frequency',), ('frequency',))
 
	def set_sweep(self, start, stop, npts):
		# update the config of the software parameter
		f = np.linspace(int(start), int(stop), num=npts)
		self.setpoints = ((f,), (f,))
		self.shapes = ((npts,), (npts,))
 
	def get(self):      
		self._instrument.write('INIT:IMM; *WAI')
		data_list = [float(v) for v in self._instrument.ask('CALC:DATA? SDAT').split(',')]
		data_arr = data_arr = np.array(data_list).reshape(len(data_list)/2,2)
		mag_array=[]
		phase_array=[]
		for comp in data_arr:
			complex_num = complex(comp[0],comp[1])
			mag_array.append(abs(complex_num))
			phase_array.append(phase(complex_num))
		return mag_array, phase_array


class ZNB20(VisaInstrument):

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

        # self.add_parameter(name='centFreq',
        #                    label='Central Frequency',
        #                    units='Hz',
        #                    get_cmd='SENS:FREQ:CENT?',
        #                    set_cmd='SENS:FREQ:CENT {:d}',
        #                    get_parser=int,
        #                    vals=vals.Numbers(1,2e10))

        # self.add_parameter(name='spanFreq',
        #                    label='Span Frequency',
        #                    units='Hz',
        #                    get_cmd='SENS:FREQ:SPAN?',
        #                    set_cmd='SENS:FREQ:SPAN {:d}',
        #                    get_parser=int,
        #                    vals=vals.Numbers(1,2e10))

		#TODO: vals make this work
        self.add_parameter(name='start', 
                           get_cmd='SENS:FREQ:START?',
                           set_cmd=self._set_start
                           )

        self.add_parameter(name='stop', 
                           get_cmd='SENS:FREQ:STOP?',
                           set_cmd=self._set_stop
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
		#TODO: check
        self.add_function('turn_on_error_tooltip', call_cmd='SYST:ERR:DISP ON')
        self.add_function('turn_off_error_tooltip', call_cmd='SYST:ERR:DISP OFF')
        self.add_function('turn_on_cont_meas', call_cmd='INIT:CONT:ALL ON')
        self.add_function('turn_off_cont_meas', call_cmd='INIT:CONT:ALL OFF')
        self.add_function('update_display_once', call_cmd='SYST:DISP:UPD ONCE')

        self.initialise()
        self.connect_message()


    def _set_start(self, val):
        self.write('SENS:FREQ:START {:.4f}'.format(val))
        self.read.set_sweep(val, self.stop(), self.npts())

    def _set_stop(self, val):
        self.write('SENS:FREQ:STOP {:.4f}'.format(val))
        self.read.set_sweep(self.start(), val, self.npts())
 
    def _set_npts(self, val):
        self.write('SENS:SWE:POIN {:.4f}'.format(val))
        self.read.set_sweep(self.start(), self.stop(), val)

    def initialise(self):
        # TODO: set input and output buffer size (its in the matlab)?
        self.write('*RST')
        self.write('SENS1:SWE:TYPE LIN')
        self.write('SENS1:SWE:TIME:AUTO ON')
        self.write('TRIG1:SEQ:SOUR IMM')
        self.write('SENS1:AVER:STAT ON')

    #TODO: get rid of this
    def getTrace(self, points):
        self.write('SENS1:AVER:STAT ON')
        self.write('AVER:CLE')
        self.write('SENS:SWE:POIN %d' %points)
        self.turn_off_cont_meas()

        avgnum = self.get('avg')
        while avgnum > 0:
            self.write('INIT:IMM; *WAI')
            avgnum-=1

        data_str = self.ask('CALC:DATA? SDAT')
        self.update_display_once()
        self.write('INIT:CONT ON')

        data_list = list(map(float, data_str.split(',')))
        data_arr = np.array(data_list).reshape(len(data_list)/2,2)

        #TODO: better python here
        mag_array=[]
        phase_array=[]
        for complex_num in data_arr:
            mag_array.append(abs(complex(complex_num[0],complex_num[1])))
            phase_array.append(phase(complex(complex_num[0],complex_num[1])))
        return [np.array(mag_array), np.array(phase_array)]
