from qcodes import VisaInstrument
from qcodes.utils.validators import Strings, Enum, Ints, MultiType, Numbers


class Keithley_2450(VisaInstrument):
    """
    QCoDeS driver for the Keithley 2450 SMU.

    NOTE:   !!! Needs to be tested !!!

    """

    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, terminator='\n', **kwargs)

        ### Sense parameters ###
        self.add_parameter('sense_mode',
                           vals=Enum('VOLT', 'CURR', 'RES'),
                           get_cmd=':SENS:FUNC?',
                           set_cmd=':SENS:FUNC "{:s}"',
                           label='Sense mode',
                           docstring='This determines whether a voltage, current or resistance is being sensed.')

        # To be tested:
        self.add_parameter('sense_value',
                           vals=float,
                           get_cmd=':READ?',
                           label='Sense value',
                           docstring='Reading the sensing value in current sense mode.')

        # To be tested:
        self.add_parameter('count',
                           vals=Ints(min_value=1, max_value=300000),
                           get_cmd=':SENS:COUN?',
                           set_cmd=':SENS:COUN {:d}',
                           label='Count',
                           docstring='The number of measurements to perform upon request.')

        # To be tested:
        self.add_parameter('average_count',
                           vals=Ints(min_value=1, max_value=100),
                           get_cmd=self._get_average_count,
                           set_cmd=self._set_average_count,
                           label='Average count',
                           docstring='The number of measurements to average over.')

        # To be tested:
        self.add_parameter('average_mode',
                           vals=Enum('MOV', 'REP'),
                           get_cmd=self._get_average_mode,
                           set_cmd=self._set_average_mode,
                           label='Average mode',
                           docstring='A moving filter will average data from sample to sample, \
                           but a true average will not be generated until the chosen count is reached. \
                           A repeating filter will only output an average once all measurement counts \
                           are collected and is hence slower.')

        # To be tested:
        self.add_parameter('average_state',
                           val_mapping={'ON': 1, 'OFF': 0},
                           get_cmd=self._get_average_state,
                           set_cmd=self._set_average_state,
                           label='Average state',
                           docstring='The state of averaging for a measurement, either on or off.')

        # To be tested:
        self.add_parameter('sense_range_auto',
                           val_mapping={'ON': 1, 'OFF': 0},
                           get_cmd=self._get_sense_range_auto,
                           set_cmd=self._set_sense_range_auto,
                           label='Sense range auto mode',
                           docstring='This determines if the range for measurements is selected manually \
                           (OFF), or automatically (ON).')

        # To be tested:
        self.add_parameter('sense_range_auto_lower_limit',
                           vals=float,
                           get_cmd=self._get_sense_range_auto_lower_limit,
                           set_cmd=self._set_sense_range_auto_lower_limit,
                           label='Auto range lower limit',
                           docstring='This sets the lower limit used when in auto-ranging mode. \
                           The lower this limit requires a longer settling time, and so you can \
                           speed up measurements by choosing a suitably high lower limit.')

        # To be tested:
        self.add_parameter('sense_range_auto_upper_limit',
                           vals=float,
                           get_cmd=self._get_sense_range_auto_upper_limit,
                           set_cmd=self._set_sense_range_auto_upper_limit,
                           label='Auto range upper limit',
                           docstring='This sets the upper limit used when in auto-ranging mode. \
                           This is only used when measuring a resistance.')

        # To be tested:
        self.add_parameter('sense_range_manual',
                           vals=float,
                           get_cmd=self._get_sense_range_manual,
                           set_cmd=self._set_sense_range_manual,
                           label='Manual range upper limit',
                           docstring='The upper limit of what is being measured when in manual mode')

    #     self.add_parameter('relative_offset',
    #                        vals=float,
    #                        get_cmd=self._get_relative_offset,
    #                        set_cmd=self._set_relative_offset,
    #                        label='Relative offset value for a measurement.',
    #                        docstring='This specifies an internal offset that can be applied to measured data')
    #
    #     self.add_parameter('relative_offset_state',
    #                        vals=Enum('OFF', 'ON') ,
    #                        get_cmd=self._get_relative_offset_state,
    #                        set_cmd=self._set_relative_offset_state,
    #                        label='Relative offset state',
    #                        docstring='This determines if the relative offset is to be applied to measurements.')
    #
    #     self.add_parameter('four_wire_mode',
    #                        vals=Enum('OFF', 'ON'),
    #                        get_cmd=self._get_four_wire_mode,
    #                        set_cmd=self._set_four_wire_mode,
    #                        label='Four-wire sensing state',
    #                        docstring='This determines whether you sense in two-wire (OFF) or \
    #                        four-wire mode (ON)')

        ### Source parameters ###
        self.add_parameter('source_mode',
                           vals=Enum('VOLT', 'CURR'),
                           get_cmd=':SOUR:FUNC?',
                           set_cmd=':SOUR:FUNC {:s}', # NOTE: self._set_mode_and_sense can be used!
                           label='Source mode',
                           docstring='This determines whether a voltage or current is being sourced.')

        # To be tested:
        self.add_parameter('source_level',
                           vals=Numbers(),
                           get_cmd=self._get_source_level,
                           set_cmd=self._set_source_level,
                           label='Source level',
                           docstring='This sets/reads the output voltage or current level of the source.')

        self.add_parameter('output_state',
                           val_mapping={'ON': 1, 'OFF': 0},
                           set_cmd=':OUTP:STAT {:d}',
                           get_cmd=':OUTP:STAT?',
                           label='Output state',
                           docstring='Determines whether output is ON or OFF.')

        self.add_parameter('source_limit',
                           vals=Numbers(),
                           get_cmd=self._get_source_limit,
                           set_cmd=self._set_source_limit,
                           label='Source limit',
                           docstring='The current (voltage) limit when sourcing voltage (current).')

        #To be tested:
        self.add_parameter('source_limit_tripped',
                           val_mapping={'YES': 1, 'NO': 0},
                           get_cmd=self._get_source_limit_tripped,
                           label='The trip state of the source limit.',
                           docstring='This reads if the source limit has been tripped during a measurement.')

        self.add_parameter('source_range',
                           vals=Numbers(),
                           get_cmd=self._get_source_range,
                           set_cmd=self._set_source_range,
                           label='Source range',
                           docstring='The voltage (current) output range when sourcing a voltage (current).')

        self.add_parameter('source_range_auto',
                           val_mapping={'ON': 1, 'OFF': 0},
                           get_cmd=self._get_source_range_auto,
                           set_cmd=self._set_source_range_auto,
                           label='Source range auto mode',
                           docstring='Determines if the range for sourcing is selected manually (OFF), \
                                     or automatically (ON).')

    #     self.add_parameter('source_delay',
    #                        vals=MultiType(float, Enum('MIN', 'DEF', 'MAX')),
    #                        get_cmd=self._get_source_delay,
    #                        set_cmd=self._set_source_delay,
    #                        label='Source measurement delay',
    #                        docstring='This determines the delay between the source changing and a measurement \
    #                        being recorded.')

    #     self.add_parameter('source_read_back',
    #                        vals=Enum('OFF', 'ON'),
    #                        get_cmd=self._get_source_read_back,
    #                        set_cmd=self._set_source_read_back,
    #                        label='Source read-back',
    #                        docstring='This determines whether the recorded output is the measured source value \
    #                        or the configured source value.')
    #
        # self.add_parameter('source_delay_auto',
        #                    vals=Enum('OFF', 'ON'),
        #                    get_cmd=self._get_source_delay_state,
        #                    set_cmd=self._set_source_delay_state,
        #                    label='',
        #                    docstring='')

    #     self.add_parameter('nplc',
    #                        get_cmd=self._get_nplc,
    #                        set_cmd=self._set_nplc,
    #                        label='Sensed input integration time',
    #                        docstring='This command sets the amount of time that the input signal is measured. \
    #                                   The amount of time is specified in parameters that are based on the \
    #                                   number of power line cycles (NPLCs). Each PLC for 60 Hz is 16.67 ms \
    #                                   (1/60) and each PLC for 50 Hz is 20 ms (1/50).')


        ### Other deprecated parameters ###
        # #deprecated
        # self.add_parameter('rangev',
        #                    get_cmd='SENS:VOLT:RANG?',
        #                    get_parser=float,
        #                    set_cmd='SOUR:VOLT:RANG {:f}',
        #                    label='Voltage range')

        # #deprecated
        # self.add_parameter('rangei',
        #                    get_cmd='SENS:CURR:RANG?',
        #                    get_parser=float,
        #                    set_cmd='SOUR:CURR:RANG {:f}',
        #                    label='Current range')

        #deprecated
        self.add_parameter('compliancev',
                           get_cmd='SENS:VOLT:PROT?',
                           get_parser=float,
                           set_cmd='SENS:VOLT:PROT {:f}',
                           label='Voltage Compliance')

        #deprecated
        self.add_parameter('compliancei',
                           get_cmd='SENS:CURR:PROT?',
                           get_parser=float,
                           set_cmd='SENS:CURR:PROT {:f}',
                           label='Current Compliance')

        # deprecated
        self.add_parameter('nplcv',
                           get_cmd='SENS:VOLT:NPLC?',
                           get_parser=float,
                           set_cmd='SENS:VOLT:NPLC {:f}',
                           label='Voltage integration time')

        #deprecated
        self.add_parameter('nplci',
                           get_cmd='SENS:CURR:NPLC?',
                           get_parser=float,
                           set_cmd='SENS:CURR:NPLC {:f}',
                           label='Current integration time')

        # deprecated
        self.add_parameter('time',
                           get_cmd=self.getTime,
                           get_parser=self._time_parser,
                           label='Relative time of measurement',
                           unit='s')

        #deprecated
        self.add_parameter('volt',
                           get_cmd=':READ?', #NOTE: self.measPosFunc can be used
                           get_parser=self._volt_parser,
                           set_cmd=':SOUR:VOLT:LEV {:.8f}',
                           label='Voltage',
                           unit='V')

        # deprecated
        self.add_parameter('voltneg',
                           get_cmd=self.measNegFunc,
                           get_parser=self._volt_parser,
                           label='Voltage',
                           unit='V')

        # deprecated
        self.add_parameter('voltzero',
                           get_cmd=self.measFunc,
                           get_parser=self._volt_parser,
                           label='Voltage',
                           unit='V')

        #deprecated
        self.add_parameter('curr',
                           get_cmd=':READ?', # NOTE: self.getCurrent can be used!
                           get_parser=self._curr_parser,
                           set_cmd=':SOUR:CURR:LEV {:.8f}',
                           label='Current',
                           unit='A')

        #deprecated
        self.add_parameter('resistance',
                           get_cmd=':READ?',
                           get_parser=self._resistance_parser,
                           label='Resistance',
                           unit='Ohm')


    ### Functions ###
    def reset(self):
        """
        Reset the instrument. When the instrument is reset, it performs the
        following actions:
            Returns the SourceMeter to the GPIB default conditions.
            Cancels all pending commands.
            Cancels all previously send `*OPC` and `*OPC?`
        """
        self.write(':*RST')

    def _get_source_level(self):
        mode = self.source_mode()
        if mode == 'VOLT':
            return self.ask(':SOUR:VOLT?')
        elif mode == 'CURR':
            return self.ask(':SOUR:CURR?')

    def _set_source_level(self, value):
        mode = self.source_mode()
        if mode == 'VOLT':
            if value<=210.0 and value>=-210.0:
                return self.write(':SOUR:VOLT {:f}'.format(value))
            else:
                raise ValueError('Out of range limits!')
        elif mode == 'CURR':
            if value <= 1.05 and value >= -1.05:
                return self.write(':SOUR:CURR {:f}'.format(value))
            else:
                raise ValueError('Out of range limits!')

    def _get_source_limit(self):
        mode = self.source_mode()
        if mode == 'VOLT':
            return self.ask(':SOUR:VOLT:ILIM?')+' A'
        elif mode == 'CURR':
            return self.ask(':SOUR:CURR:VLIM?')+' V'

    def _set_source_limit(self, value):
        mode = self.source_mode()
        if mode == 'VOLT':
            if value<=1.05 and value>=-1.05:
                return self.write(':SOUR:VOLT:ILIM {:f}'.format(value))
            else:
                raise ValueError('Out of range limits!')
        elif mode == 'CURR':
            if value<=210.0 and value>=-210.0:
                return self.write(':SOUR:CURR:VLIM {:f}'.format(value))
            else:
                raise ValueError('Out of range limits!')

    def _get_source_limit_tripped(self):
        mode = self.source_mode()
        if mode == 'VOLT':
            return self.ask(':SOUR:VOLT:ILIM:TRIP?')
        elif mode == 'CURR':
            return self.ask(':SOUR:CURR:VLIM:TRIP?')

    def _get_source_range(self):
        mode = self.source_mode()
        if mode == 'VOLT':
            return self.ask(':SOUR:VOLT:RANG?')+' V'
        elif mode == 'CURR':
            return self.ask(':SOUR:CURR:RANG?')+' A'

    def _set_source_range(self, value):
        mode = self.source_mode()
        if mode == 'VOLT':
            if value<=200.0 and value>=-200.0:
                return self.write(':SOUR:VOLT:RANG {:f}'.format(value))
            else:
                raise ValueError('Out of range limits!')
        elif mode == 'CURR':
            if value<=1.0 and value>=-1.0:
                return self.write(':SOUR:CURR:RANG {:f}'.format(value))
            else:
                raise ValueError('Out of range limits!')

    def _get_source_range_auto(self):
        mode = self.source_mode()
        if mode == 'VOLT':
            return self.ask(':SOUR:VOLT:RANG:AUTO?')
        elif mode == 'CURR':
            return self.ask(':SOUR:CURR:RANG:AUTO?')

    def _set_source_range_auto(self, value):
        mode = self.source_mode()
        if mode == 'VOLT':
            return self.write(':SOUR:VOLT:RANG:AUTO {:d}'.format(value))
        elif mode == 'CURR':
            return self.write(':SOUR:CURR:RANG:AUTO {:d}'.format(value))

    def _get_average_count(self):
        mode = self.sense_mode()
        if mode == 'VOLT':
            return self.ask(':SENS:VOLT:AVER:COUN?')
        elif mode == 'CURR':
            return self.ask(':SENS:CURR:AVER:COUN?')
        elif mode == 'RES':
            return self.ask(':SENS:RES:AVER:COUN?')

    def _set_average_count(self, value):
        mode = self.sense_mode()
        if mode == 'VOLT':
            return self.ask(':SENS:VOLT:AVER:COUN {:d}'.format(value))
        elif mode == 'CURR':
            return self.ask(':SENS:CURR:AVER:COUN {:d}'.format(value))
        elif mode == 'RES':
            return self.ask(':SENS:RES:AVER:COUN {:d}'.format(value))

    def _get_average_mode(self):
        mode = self.sense_mode()
        if mode == 'VOLT':
            return self.ask(':SENS:VOLT:AVER:TCON?')
        elif mode == 'CURR':
            return self.ask(':SENS:CURR:AVER:TCON?')
        elif mode == 'RES':
            return self.ask(':SENS:RES:AVER:TCON?')

    def _set_average_mode(self, filter_type):
        mode = self.sense_mode()
        if mode == 'VOLT':
            return self.ask(':SENS:VOLT:AVER:TCON {:s}'.format(filter_type))
        elif mode == 'CURR':
            return self.ask(':SENS:CURR:AVER:TCON {:s}'.format(filter_type))
        elif mode == 'RES':
            return self.ask(':SENS:RES:AVER:TCON {:s}'.format(filter_type))

    def _get_average_state(self):
        mode = self.sense_mode()
        if mode == 'VOLT':
            return self.ask(':SENS:VOLT:AVER?')
        elif mode == 'CURR':
            return self.ask(':SENS:CURR:AVER?')
        elif mode == 'RES':
            return self.ask(':SENS:RES:AVER?')

    def _set_average_state(self, value):
        mode = self.sense_mode()
        if mode == 'VOLT':
            return self.ask(':SENS:VOLT:AVER {:d}'.format(value))
        elif mode == 'CURR':
            return self.ask(':SENS:CURR:AVER {:d}'.format(value))
        elif mode == 'RES':
            return self.ask(':SENS:RES:AVER {:d}'.format(value))

    def _get_sense_range_auto(self):
        mode = self.sense_mode()
        if mode == 'VOLT':
            return self.ask(':SENS:VOLT:RANG:AUTO?')
        elif mode == 'CURR':
            return self.ask(':SENS:CURR:RANG:AUTO?')
        elif mode == 'RES':
            return self.ask(':SENS:RES:RANG:AUTO?')

    def _set_sense_range_auto(self, value):
        mode = self.sense_mode()
        if mode == 'VOLT':
            return self.write(':SENS:VOLT:RANG:AUTO {:d}'.format(value))
        elif mode == 'CURR':
            return self.write(':SENS:CURR:RANG:AUTO {:d}'.format(value))
        elif mode == 'RES':
            return self.ask(':SENS:RES:RANG:AUTO {:d}'.format(value))

    def _get_sense_range(self):
        mode = self.sense_mode()
        if mode == 'VOLT':
            return self.ask(':SENS:VOLT:RANG?')+' V'
        elif mode == 'CURR':
            return self.ask(':SENS:CURR:RANG?')+' A'
        elif mode == 'RES':
            return self.ask(':SENS:RES:RANG?')+' Ohms'

    def _set_sense_range(self, value):
        mode = self.sense_mode()
        if mode == 'VOLT':
            if value<=200.0 and value>=0.02:
                return self.write(':SENS:VOLT:RANG {:f}'.format(value))
            else:
                raise ValueError('Out of range limits!')
        elif mode == 'CURR':
            if value<=1.0 and value>=1e-8:
                return self.write(':SENS:CURR:RANG {:f}'.format(value))
            else:
                raise ValueError('Out of range limits!')
        elif mode == 'RES':
            if value <= 2e8 and value >= 20:
                return self.write(':SENS:RES:RANG {:f}'.format(value))
            else:
                raise ValueError('Out of range limits!')

    def _get_sense_range_auto_lower_limit(self):
        mode = self.sense_mode()
        if mode == 'VOLT':
            return self.ask(':SENS:VOLT:RANG:AUTO:LLIM?')+' V'
        elif mode == 'CURR':
            return self.ask(':SENS:CURR:RANG:AUTO:LLIM?')+' A'
        elif mode == 'RES':
            return self.ask(':SENS:RES:RANG:AUTO:LLIM?')+' Ohms'

    def _set_sense_range_auto_lower_limit(self, value):
        mode = self.sense_mode()
        if mode == 'VOLT':
            if value<=200.0 and value>=0.02:
                return self.write(':SENS:VOLT:RANG:AUTO:LLIM {:f}'.format(value))
            else:
                raise ValueError('Out of range limits!')
        elif mode == 'CURR':
            if value<=1.0 and value>=1e-8:
                return self.write(':SENS:CURR:RANG:AUTO:LLIM {:f}'.format(value))
            else:
                raise ValueError('Out of range limits!')
        elif mode == 'RES':
            if value <= 2e8 and value >= 2:
                return self.write(':SENS:RES:RANG:AUTO:LLIM {:f}'.format(value))
            else:
                raise ValueError('Out of range limits!')

    def _get_sense_range_auto_upper_limit(self):
        mode = self.sense_mode()
        if mode == 'VOLT':
            return self.ask(':SENS:VOLT:RANG:AUTO:ULIM?')+' V'
        elif mode == 'CURR':
            raise ValueError('Wrong sense mode for auto range upper limit!')
        elif mode == 'RES':
            return self.ask(':SENS:RES:RANG:AUTO:ULIM?')+' Ohms'

    def _set_sense_range_auto_upper_limit(self, value):
        mode = self.sense_mode()
        if mode == 'VOLT':
            raise ValueError('Auto range upper limit can be set only for resistance!')
        elif mode == 'CURR':
            raise ValueError('Auto range upper limit can be set only for resistance!')
        elif mode == 'RES':
            lower_limit = self.sense_range_auto_lower_limit()
            if value <= 2e8 and value >= 20 and lower_limit <= value:
                return self.write(':SENS:RES:RANG:AUTO:ULIM {:f}'.format(value))
            else:
                raise ValueError('Out of range limits!')

    ### Other deprecated functions ###
    # deprecated
    def _source_mode(self):
        """
        This helper function is used to manage most settable parameters to ensure the device is
        consistently configured for the correct output mode.
        """
        mode = self.source_mode().get_latest()

        if mode is not None:
            return mode
        else:
            return self.source_mode()

    # deprecated
    def _sense_mode(self):
        """
        This helper function is used to manage most settable parameters to ensure the device is
        consistently configured for the correct sensing mode.
        """
        mode = self.sense_mode().get_latest()

        if mode is not None:
            return mode
        else:
            return self.sense_mode()

    # deprecated
    def _set_mode_and_sense(self, msg):
        # This helps set the correct read out curr/volt configuration
        if msg == 'VOLT':
            self.sense('CURR')
        elif msg == 'CURR':
            self.sense('VOLT')
        else:
            raise AttributeError('Mode does not exist')
        self.write(':SOUR:FUNC {:s}'.format(msg))

    # deprecated
    def _volt_parser(self, msg):
        fields = [float(x) for x in msg.split(',')]
        return fields[0]

    # deprecated
    def _curr_parser(self, msg):
        fields = [float(x) for x in msg.split(',')]
        return fields[1]

    # deprecated
    def _resistance_parser(self, msg):
        fields = [float(x) for x in msg.split(',')]
        return fields[0]/fields[1]

    # deprecated
    def _time_parser(self, msg):
        fields = [float(x) for x in msg.split(',')]
        return fields[2]

    # deprecated
    def setNPLC(self,n):
        self.write(':SENSe:VOLTage:NPLCycles '+str(n))

    # deprecated
    def makeBuffer(self):
        self.write('TRACe:MAKE "MykhBuffer1", 20')

    # deprecated
    def clearBuffer(self):
        self.write(':TRACe:CLEar "MykhBuffer1"')

    # deprecated
    def measNegFunc(self):
        self.write('SOURce:CURR -0.02')
        self.write('OUTput ON')
        self.write('TRACe:TRIGger "MykhBuffer1"')
        return self.ask('TRACe:DATA? 1, 1, "MykhBuffer1", SOUR, READ, SEC')

    # deprecated
    def measFunc(self):
        self.write('OUTput ON')
        self.write('TRACe:TRIGger "MykhBuffer1"')
        return self.ask('TRACe:DATA? 1, 1, "MykhBuffer1", SOUR, READ, SEC')

    # deprecated
    def getTime(self):
        return self.ask('TRACe:DATA? 1, 1, "MykhBuffer1", SOUR, READ, SEC')

    # deprecated
    def getCurrent(self):
        return self.ask('TRACe:DATA? 1, 1, "MykhBuffer1", SOUR, READ, SEC')

    # deprecated
    def measPosFunc(self):
        self.write('SOURce:CURR 0.02')
        self.write('OUTput ON')
        self.write('TRACe:TRIGger "MykhBuffer1"')
        return self.ask('TRACe:DATA? 1, 1, "MykhBuffer1", SOUR, READ, SEC')

    # deprecated
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