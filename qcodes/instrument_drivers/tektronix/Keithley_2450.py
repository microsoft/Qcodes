# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 10:08:44 2018

@author: LabUSer
"""

from qcodes import VisaInstrument
from qcodes.utils.validators import Strings, Enum


class Keithley_2450(VisaInstrument):
    """
    QCoDeS driver for the Keithley 2450 voltage source.
    """
    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, terminator='\n', **kwargs)

        self.add_parameter('rangev',
                           get_cmd='SENS:VOLT:RANG?',
                           get_parser=float,
                           set_cmd='SOUR:VOLT:RANG {:f}',
                           label='Voltage range')

        self.add_parameter('rangei',
                           get_cmd='SENS:CURR:RANG?',
                           get_parser=float,
                           set_cmd='SOUR:CURR:RANG {:f}',
                           label='Current range')

        self.add_parameter('compliancev',
                           get_cmd='SENS:VOLT:PROT?',
                           get_parser=float,
                           set_cmd='SENS:VOLT:PROT {:f}',
                           label='Voltage Compliance')

        self.add_parameter('compliancei',
                           get_cmd='SENS:CURR:PROT?',
                           get_parser=float,
                           set_cmd='SENS:CURR:PROT {:f}',
                           label='Current Compliance')
#
#        self.add_parameter('volt',
#                           get_cmd=':READ?',
#                           get_parser=self._volt_parser,
#                           set_cmd=':SOUR:VOLT:LEV {:.8f}',
#                           label='Voltage',
#                           unit='V')

        self.add_parameter('curr',
                           get_cmd=self.getCurrent,
                           get_parser=self._curr_parser,
                           set_cmd=':SOUR:CURR:LEV {:.8f}',
                           label='Current',
                           unit='A')

        self.add_parameter('mode',
                           vals=Enum('VOLT', 'CURR'),
                           get_cmd=':SOUR:FUNC?',
                           set_cmd=self._set_mode_and_sense,
                           label='Mode')

        self.add_parameter('sense',
                           vals=Strings(),
                           get_cmd=':SENS:FUNC?',
                           set_cmd=':SENS:FUNC "{:s}"',
                           label='Sense mode')

        self.add_parameter('output',
                           get_parser=int,
                           set_cmd=':OUTP:STAT {:d}',
                           get_cmd=':OUTP:STAT?')

        self.add_parameter('nplcv',
                           get_cmd='SENS:VOLT:NPLC?',
                           get_parser=float,
                           set_cmd='SENS:VOLT:NPLC {:f}',
                           label='Voltage integration time')

        self.add_parameter('nplci',
                           get_cmd='SENS:CURR:NPLC?',
                           get_parser=float,
                           set_cmd='SENS:CURR:NPLC {:f}',
                           label='Current integration time')

        self.add_parameter('time',
                           get_cmd=self.getTime,
                           get_parser=self._time_parser,
                           label='Relative time of measurement',
                           unit='s')

        self.add_parameter('volt',
                           get_cmd=self.measPosFunc,
                           get_parser=self._volt_parser,
                           label='Voltage',
                           unit='V')
        
        self.add_parameter('voltneg',
                           get_cmd=self.measNegFunc,
                           get_parser=self._volt_parser,
                           label='Voltage',
                           unit='V')
        
        self.add_parameter('voltzero',
                           get_cmd=self.measFunc,
                           get_parser=self._volt_parser,
                           label='Voltage',
                           unit='V')
        
        self.connect_message()
        
    def _set_mode_and_sense(self, msg):
        # This helps set the correct read out curr/volt
        if msg == 'VOLT':
            self.sense('CURR')
        elif msg == 'CURR':
            self.sense('VOLT')
        else:
            raise AttributeError('Mode does not exist')
        self.write(':SOUR:FUNC {:s}'.format(msg))

    def reset(self):
        """
        Reset the instrument. When the instrument is reset, it performs the
        following actions.

            Returns the SourceMeter to the GPIB default conditions.

            Cancels all pending commands.

            Cancels all previously send `*OPC` and `*OPC?`
        """
        self.write(':*RST')

    def _volt_parser(self, msg):
        fields = [float(x) for x in msg.split(',')]
        return fields[1]

    def _curr_parser(self, msg):
        fields = [float(x) for x in msg.split(',')]
        return fields[0]

    def _time_parser(self, msg):
        fields = [float(x) for x in msg.split(',')]
        return fields[2]

    def setVoltSens(self):
        self.write('*RST') 
        self.write(':ROUT:TERM REAR')
        self.write('SENSe:FUNCtion "VOLT"') 
        self.write('SENSe:VOLTage:RANGe:AUTO ON') 
        self.write('SENSe:VOLTage:UNIT VOLT') 
        self.write('SENSe:VOLTage:RSENse ON')
        self.write('SOURce:FUNCtion CURR') 
        self.write('SOURce:CURR 0.02') 
        self.write('SOURce:CURR:VLIM 2')  
        self.write('SENSe:COUNT 1')
        self.write(':SENSe:VOLTage:NPLCycles 10')
        self.write(':DISPlay:VOLTage:DIGits 6')
     
        
    def setCurrent(self, i):
        self.write('SOURce:CURR '+str(i)) 
        
    def setNPLC(self,n): 
        self.write(':SENSe:VOLTage:NPLCycles '+str(n))
    
    def setNegCurr(self):
        self.write('SOURce:CURR -0.02')
        
    def measPosFunc(self):
        self.write('SOURce:CURR 0.02')
        self.write('OUTput ON')
        self.write('TRACe:TRIGger "MykhBuffer1"')
        return self.ask('TRACe:DATA? 1, 1, "MykhBuffer1", SOUR, READ, SEC')
        
    def setOutputOFF(self):
        self.write('OUTput OFF')
        
    def setOutputON(self):
        self.write('OUTput ON')
        
    def makeBuffer(self):
        self.write('TRACe:MAKE "MykhBuffer1", 20')
        
    def clearBuffer(self):
        self.write(':TRACe:CLEar "MykhBuffer1"')
 
    def measNegFunc(self):
        self.write('SOURce:CURR -0.02')
        self.write('OUTput ON')
        self.write('TRACe:TRIGger "MykhBuffer1"')
        return self.ask('TRACe:DATA? 1, 1, "MykhBuffer1", SOUR, READ, SEC')
    
    def measFunc(self):
        self.write('OUTput ON')
        self.write('TRACe:TRIGger "MykhBuffer1"')
        return self.ask('TRACe:DATA? 1, 1, "MykhBuffer1", SOUR, READ, SEC')
    
    def getTime(self):
        return self.ask('TRACe:DATA? 1, 1, "MykhBuffer1", SOUR, READ, SEC')
    
    def getCurrent(self):
        return self.ask('TRACe:DATA? 1, 1, "MykhBuffer1", SOUR, READ, SEC')
        
    