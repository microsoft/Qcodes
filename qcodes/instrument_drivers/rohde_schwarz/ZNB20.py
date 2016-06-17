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
                           set_cmd='AVER:COUN? {:d}',
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

        self.initialise()
        self.connect_message()

    def initialise(self):
        # TODO: set input and output buffer size (its in the matlab)?
        self.write('SENS1:SWE:TYPE LIN')
        self.write('SENS1:SWE:TIME:AUTO ON')
        self.write('TRIG1:SEQ:SOUR IMM')
        # need to set averages?
        self.write('SENS1:AVER:STAT ON')

    def getTrace(self):
      self.write('SENS:SWE:POIN'+points)
      self.write('INIT:CONT OFF') 
      self.write('AVER:CLE')

      avgnum = self.avg
      while (avgnum > 0):
          self.write('INIT')
          self.ask('*OPC?')

      data_str = self.ask('CALC:DATA? SDAT')
      self.write('INIT:CONT ON')

      data_list = list(map(float, data_str.split()))
      data_mat = list(zip(data_list[0::2], data_list[1::2]))

      traces = []
      for (re,im) in data_mat:
          mag1 = abs(complex(re, im))
          phase1 = phase(complex(re, im))
          print(mag1, phase1)
          traces.append([mag1, phase1])
        return traces
