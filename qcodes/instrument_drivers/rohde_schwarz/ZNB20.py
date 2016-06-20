from qcodes import VisaInstrument
from qcodes.utils import validators as vals
from cmath import phase

'''
    TODO: 
        better error messages
        add functionality (ability to set start and stop freq as well as centre and span)
        needs on/off status?
        write string to int and do it
'''

class RohdeSchwarz_ZNB20(VisaInstrument):

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

        self.add_parameter(name='centFreq',
                           label='Central Frequency',
                           units='Hz',
                           get_cmd='SENS:FREQ:CENT?',
                           set_cmd='SENS:FREQ:CENT {:d}',
                           get_parser=int,
                           vals=vals.Numbers(1,2e10))

        self.add_parameter(name='spanFreq',
                           label='Span Frequency',
                           units='Hz',
                           get_cmd='SENS:FREQ:SPAN?',
                           set_cmd='SENS:FREQ:SPAN {:d}',
                           get_parser=int,
                           vals=vals.Numbers(1,2e10))

        # TODO: add params (trigger source, sweep time auto, )

        self.add_function('reset', call_cmd='*RST')
        self.add_function('turn_on_error_tooltip', call_cmd='SYST:ERR:DISP:REM ON')
        self.add_function('turn_off_error_tooltip', call_cmd='SYST:ERR:DISP:REM OFF')
        self.add_function('turn_off_cont_meas', call_cmd='INIT:CONT:ALL OFF')

        self.initialise()
        self.connect_message()

    def initialise(self):
        # TODO: set input and output buffer size (its in the matlab)?
        self.write('*RST')
        self.write('SENS1:SWE:TYPE LIN')
        self.write('SENS1:SWE:TIME:AUTO ON')
        self.write('TRIG1:SEQ:SOUR IMM')
        # need to set averages?
        self.write('SENS1:AVER:STAT ON')

    # TODO: use *WAI or *OPC?
    # TODO: what does repeating and decreasing avgnumber do?

    def getTraceSimple(self,points):
        self.write('SENS1:AVER:STAT OFF')
        self.write('INIT:CONT:ALL OFF') 
        self.write('SENS:SWE:POIN %d' % points)
        data = self.ask('CALC:DATA? SDAT')
        self.write('SYST:DISPl:UPD ONCE')
        return data

    def getTrace(self, points):
        self.write('SENS1:AVER:STAT ON')
        self.write('AVER:CLE')
        self.write('SENS:SWE:POIN %d' %points)
        self.write('INIT:CONT:ALL OFF') 
        self.write('INIT:IMM; *WAI')
        data_str = self.ask('CALC:DATA? SDAT')
        self.write('SYST:DISPl:UPD ONCE')


        # while (self.get('avg') > 0):
        #     self.write('INIT:IMM; *WAI')
        #     self.set('avg') = self.get('avg')-1

        # data_str = self.ask('CALC:DATA? SDAT')
        # self.write('INIT:CONT ON')

        data_list = list(map(float, data_str.split(',')))
        data_mat = list(zip(data_list[0::2], data_list[1::2]))

        traces = []
        for (re,im) in data_mat:
            mag1 = abs(complex(re, im))
            phase1 = phase(complex(re, im))
            traces.append([mag1, phase1])
        return traces
