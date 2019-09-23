from qcodes import VisaInstrument
from qcodes.utils.validators import Strings, Enum, Ints, MultiType, Numbers


class Keithley_2450(VisaInstrument):
    """
    QCoDeS driver for the Keithley 2450 SMU.

    Written/modified by R.Savytskyy and M.Johnson (23/09/2019)

    NOTE: Not full list of parameters, however basic functions are implemented.
          Needs further testing, but is ready for usage.
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

        self.add_parameter('sense_value',
                           vals=Numbers(),
                           get_cmd=':READ?',
                           label='Sense value',
                           docstring='Reading the sensing value of the active sense mode.')

        self.add_parameter('count',
                           vals=Ints(min_value=1, max_value=300000),
                           get_cmd=':SENS:COUN?',
                           set_cmd=':SENS:COUN {:d}',
                           label='Count',
                           docstring='The number of measurements to perform upon request.')

        self.add_parameter('average_count',
                           vals=MultiType(Ints(min_value=1, max_value=100),
                                          Enum('MIN', 'DEF', 'MAX')),
                           get_cmd=self._get_average_count,
                           set_cmd=self._set_average_count,
                           label='Average count',
                           docstring='The number of measurements to average over.')

        self.add_parameter('average_mode',
                           vals=Enum('MOV', 'REP'),
                           get_cmd=self._get_average_mode,
                           set_cmd=self._set_average_mode,
                           label='Average mode',
                           docstring='A moving filter will average data from sample to sample, \
                           but a true average will not be generated until the chosen count is reached. \
                           A repeating filter will only output an average once all measurement counts \
                           are collected and is hence slower.')

        self.add_parameter('average_state',
                           val_mapping={'ON': 1, 'OFF': 0},
                           get_cmd=self._get_average_state,
                           set_cmd=self._set_average_state,
                           label='Average state',
                           docstring='The state of averaging for a measurement, either on or off.')

        self.add_parameter('sense_range_auto',
                           val_mapping={'ON': 1, 'OFF': 0},
                           get_cmd=self._get_sense_range_auto,
                           set_cmd=self._set_sense_range_auto,
                           label='Sense range auto mode',
                           docstring='This determines if the range for measurements is selected manually \
                           (OFF), or automatically (ON).')

        self.add_parameter('sense_range_auto_lower_limit',
                           vals=Numbers(),
                           get_cmd=self._get_sense_range_auto_lower_limit,
                           set_cmd=self._set_sense_range_auto_lower_limit,
                           label='Auto range lower limit',
                           docstring='This sets the lower limit used when in auto-ranging mode. \
                           The lower this limit requires a longer settling time, and so you can \
                           speed up measurements by choosing a suitably high lower limit.')

        self.add_parameter('sense_range_auto_upper_limit',
                           vals=Numbers(),
                           get_cmd=self._get_sense_range_auto_upper_limit,
                           set_cmd=self._set_sense_range_auto_upper_limit,
                           label='Auto range upper limit',
                           docstring='This sets the upper limit used when in auto-ranging mode. \
                           This is only used when measuring a resistance.')

        # TODO: needs connection with source range setting
        self.add_parameter('sense_range_manual',
                           vals=Numbers(),
                           get_cmd=self._get_sense_range_manual,
                           set_cmd=self._set_sense_range_manual,
                           label='Manual range upper limit',
                           docstring='The upper limit of what is being measured when in manual mode')

        self.add_parameter('nplc',
                           vals=Numbers(min_value=0.01, max_value=10),
                           get_cmd=self._get_nplc,
                           set_cmd=self._set_nplc,
                           label='Sensed input integration time',
                           docstring='This command sets the amount of time that the input signal is measured. \
                                      The amount of time is specified in parameters that are based on the \
                                      number of power line cycles (NPLCs). Each PLC for 60 Hz is 16.67 ms \
                                      (1/60) and each PLC for 50 Hz is 20 ms (1/50).')

        self.add_parameter('relative_offset',
                           vals=Numbers(),
                           get_cmd=self._get_relative_offset,
                           set_cmd=self._set_relative_offset,
                           label='Relative offset value for a measurement.',
                           docstring='This specifies an internal offset that can be applied to measured data')

        self.add_parameter('relative_offset_state',
                           val_mapping={'ON': 1, 'OFF': 0},
                           get_cmd=self._get_relative_offset_state,
                           set_cmd=self._set_relative_offset_state,
                           label='Relative offset state',
                           docstring='This determines if the relative offset is to be applied to measurements.')

        self.add_parameter('four_wire_mode',
                           val_mapping={'ON': 1, 'OFF': 0},
                           get_cmd=self._get_four_wire_mode,
                           set_cmd=self._set_four_wire_mode,
                           label='Four-wire sensing state',
                           docstring='This determines whether you sense in two-wire (OFF) or \
                           four-wire mode (ON)')

        ### Source parameters ###
        self.add_parameter('source_mode',
                           vals=Enum('VOLT', 'CURR'),
                           get_cmd=':SOUR:FUNC?',
                           set_cmd=':SOUR:FUNC {:s}',
                           label='Source mode',
                           docstring='This determines whether a voltage or current is being sourced.')

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

        self.add_parameter('source_read_back',
                           val_mapping={'ON': 1, 'OFF': 0},
                           get_cmd=self._get_source_read_back,
                           set_cmd=self._set_source_read_back,
                           label='Source read-back',
                           docstring='This determines whether the recorded output is the measured source value \
                           or the configured source value. The former increases the precision, \
                           but slows down the measurements.')

        # Note: delay value for 'MAX' is 10 000 instead of 4.
        self.add_parameter('source_delay',
                           vals=MultiType(Numbers(min_value=0.0, max_value=4.0),
                                          Enum('MIN', 'DEF', 'MAX')),
                           get_cmd=self._get_source_delay,
                           set_cmd=self._set_source_delay,
                           label='Source measurement delay',
                           docstring='This determines the delay between the source changing and a measurement \
                           being recorded.')

        self.add_parameter('source_delay_auto',
                           val_mapping={'ON': 1, 'OFF': 0},
                           get_cmd=self._get_source_delay_auto_state,
                           set_cmd=self._set_source_delay_auto_state,
                           label='Source measurement delay auto state',
                           docstring='This determines the autodelay between the source changing and a measurement \
                           being recorded set to state ON/OFF.')

        self.add_parameter('source_overvoltage_protection',
                           vals=Enum('PROT2', 'PROT5', 'PROT10', 'PROT20', 'PROT40', 'PROT60', 'PROT80', 'PROT100',
                                     'PROT120', 'PROT140', 'PROT160', 'PROT180', 'NONE'),
                           get_cmd='SOUR:VOLT:PROT?',
                           set_cmd='SOUR:VOLT:PROT {:s}',
                           label='Source overvoltage protection',
                           docstring='This sets the overvoltage protection setting of the source output. \
                           Overvoltage protection restricts the maximum voltage level that the instrument can source. \
                           It is in effect when either current or voltage is sourced.')

        self.add_parameter('source_overvoltage_protection_tripped',
                           val_mapping={'True': 1, 'False': 0},
                           get_cmd='SOUR:VOLT:PROT:TRIP?',
                           label='Source overvoltage protection tripped status',
                           docstring='If the voltage source does not exceed the set protection limits, the return is 0. \
                           If the voltage source exceeds the set limits, the return is 1.')


        ### Other deprecated parameters ###
        # deprecated
        self.add_parameter('volt',
                           get_cmd=':READ?',
                           set_cmd=':SOUR:VOLT:LEV {:.8f}',
                           label='Voltage',
                           unit='V')

        # deprecated
        self.add_parameter('curr',
                           get_cmd=':READ?',
                           set_cmd=':SOUR:CURR:LEV {:.8f}',
                           label='Current',
                           unit='A')

        # deprecated
        self.add_parameter('resistance',
                           get_cmd=':READ?',
                           label='Resistance',
                           unit='Ohm')

        # deprecated
        # self.add_parameter('voltneg',
        #                    get_cmd=self.measNegFunc,
        #                    get_parser=self._volt_parser,
        #                    label='Voltage',
        #                    unit='V')

        # deprecated
        # self.add_parameter('voltzero',
        #                    get_cmd=self.measFunc,
        #                    get_parser=self._volt_parser,
        #                    label='Voltage',
        #                    unit='V')

        # deprecated
        # self.add_parameter('time',
        #                    get_cmd=self.getTime,
        #                    get_parser=self._time_parser,
        #                    label='Relative time of measurement',
        #                    unit='s')


    ### Functions ###
    def reset(self):
        """
        Resets the instrument. During reset, it cancels all pending commands
        and all previously sent `*OPC` and `*OPC?`
        """
        self.write(':*RST')

    def _get_source_level(self):
        mode = self.source_mode()
        if 'VOLT' in mode:
            return self.ask(':SOUR:VOLT?')
        elif 'CURR' in mode:
            return self.ask(':SOUR:CURR?')
        else:
            raise UserWarning('Unknown source mode')

    def _set_source_level(self, value):
        mode = self.source_mode()
        if 'VOLT' in mode:
            if value>=-210.0 and value<=210.0:
                return self.write(':SOUR:VOLT {:f}'.format(value))
            else:
                raise ValueError('Out of range limits!')
        elif 'CURR' in mode:
            if value >= -1.05 and value <= 1.05:
                return self.write(':SOUR:CURR {:f}'.format(value))
            else:
                raise ValueError('Out of range limits!')
        else:
            raise UserWarning('Unknown source mode')

    def _get_source_limit(self):
        mode = self.source_mode()
        if 'VOLT' in mode:
            return self.ask(':SOUR:VOLT:ILIM?')+' A'
        elif 'CURR' in mode:
            return self.ask(':SOUR:CURR:VLIM?')+' V'
        else:
            raise UserWarning('Unknown source mode')

    def _set_source_limit(self, value):
        mode = self.source_mode()
        if 'VOLT' in mode:
            if value>=-1.05 and value<=1.05:
                return self.write(':SOUR:VOLT:ILIM {:f}'.format(value))
            else:
                raise ValueError('Out of range limits!')
        elif 'CURR' in mode:
            if value>=-210.0 and value<=210.0:
                return self.write(':SOUR:CURR:VLIM {:f}'.format(value))
            else:
                raise ValueError('Out of range limits!')
        else:
            raise UserWarning('Unknown source mode')

    def _get_source_limit_tripped(self):
        mode = self.source_mode()
        if 'VOLT' in mode:
            return self.ask(':SOUR:VOLT:ILIM:TRIP?')
        elif 'CURR' in mode:
            return self.ask(':SOUR:CURR:VLIM:TRIP?')
        else:
            raise UserWarning('Unknown source mode')

    def _get_source_range(self):
        mode = self.source_mode()
        if 'VOLT' in mode:
            return self.ask(':SOUR:VOLT:RANG?')+' V'
        elif 'CURR' in mode:
            return self.ask(':SOUR:CURR:RANG?')+' A'
        else:
            raise UserWarning('Unknown source mode')

    def _set_source_range(self, value):
        mode = self.source_mode()
        if 'VOLT' in mode:
            if value>=-200.0 and value<=200.0:
                return self.write(':SOUR:VOLT:RANG {:f}'.format(value))
            else:
                raise ValueError('Out of range limits!')
        elif 'CURR' in mode:
            if value>=-1.0 and value<=1.0:
                return self.write(':SOUR:CURR:RANG {:f}'.format(value))
            else:
                raise ValueError('Out of range limits!')
        else:
            raise UserWarning('Unknown source mode')

    def _get_source_range_auto(self):
        mode = self.source_mode()
        if 'VOLT' in mode:
            return self.ask(':SOUR:VOLT:RANG:AUTO?')
        elif 'CURR' in mode:
            return self.ask(':SOUR:CURR:RANG:AUTO?')
        else:
            raise UserWarning('Unknown source mode')

    def _set_source_range_auto(self, value):
        mode = self.source_mode()
        if 'VOLT' in mode:
            return self.write(':SOUR:VOLT:RANG:AUTO {:d}'.format(value))
        elif 'CURR' in mode:
            return self.write(':SOUR:CURR:RANG:AUTO {:d}'.format(value))
        else:
            raise UserWarning('Unknown source mode')

    def _get_source_read_back(self):
        mode = self.source_mode()
        if 'VOLT' in mode:
            return self.ask(':SOUR:VOLT:READ:BACK?')
        elif 'CURR' in mode:
            return self.ask(':SOUR:CURR:READ:BACK?')
        else:
            raise UserWarning('Unknown source mode')

    def _set_source_read_back(self, value):
        mode = self.source_mode()
        if 'VOLT' in mode:
            return self.write(':SOUR:VOLT:READ:BACK {:d}'.format(value))
        elif 'CURR' in mode:
            return self.write(':SOUR:CURR:READ:BACK {:d}'.format(value))
        else:
            raise UserWarning('Unknown source mode')

    def _get_source_delay(self):
        mode = self.source_mode()
        if 'VOLT' in mode:
            return self.ask(':SOUR:VOLT:DEL?')
        elif 'CURR' in mode:
            return self.ask(':SOUR:CURR:DEL?')
        else:
            raise UserWarning('Unknown source mode')

    def _set_source_delay(self, value):
        mode = self.source_mode()
        if 'VOLT' in mode:
            return self.write(':SOUR:VOLT:DEL {}'.format(value))
        elif 'CURR' in mode:
            return self.write(':SOUR:CURR:DEL {}'.format(value))
        else:
            raise UserWarning('Unknown source mode')

    def _get_source_delay_auto_state(self):
        mode = self.source_mode()
        if 'VOLT' in mode:
            return self.ask(':SOUR:VOLT:DEL:AUTO?')
        elif 'CURR' in mode:
            return self.ask(':SOUR:CURR:DEL:AUTO?')
        else:
            raise UserWarning('Unknown source mode')

    def _set_source_delay_auto_state(self, value):
        mode = self.source_mode()
        if 'VOLT' in mode:
            return self.write(':SOUR:VOLT:DEL:AUTO {:d}'.format(value))
        elif 'CURR' in mode:
            return self.write(':SOUR:CURR:DEL:AUTO {:d}'.format(value))
        else:
            raise UserWarning('Unknown source mode')

    def _get_average_count(self):
        mode = self.sense_mode()
        if 'VOLT' in mode:
            return self.ask(':SENS:VOLT:AVER:COUNT?')
        elif 'CURR' in mode:
            return self.ask(':SENS:CURR:AVER:COUNT?')
        elif 'RES' in mode:
            return self.ask(':SENS:RES:AVER:COUNT?')
        else:
            raise UserWarning('Unknown sense mode')

    def _set_average_count(self, value):
        mode = self.sense_mode()
        if 'VOLT' in mode:
            return self.write(':SENS:VOLT:AVER:COUNT {}'.format(value))
        elif 'CURR' in mode:
            return self.write(':SENS:CURR:AVER:COUNT {}'.format(value))
        elif 'RES' in mode:
            return self.write(':SENS:RES:AVER:COUNT {}'.format(value))
        else:
            raise UserWarning('Unknown sense mode')

    def _get_average_mode(self):
        mode = self.sense_mode()
        if 'VOLT' in mode:
            return self.ask(':SENS:VOLT:AVER:TCON?')
        elif 'CURR' in mode:
            return self.ask(':SENS:CURR:AVER:TCON?')
        elif 'RES' in mode:
            return self.ask(':SENS:RES:AVER:TCON?')
        else:
            raise UserWarning('Unknown sense mode')

    def _set_average_mode(self, filter_type):
        mode = self.sense_mode()
        if 'VOLT' in mode:
            return self.write(':SENS:VOLT:AVER:TCON {:s}'.format(filter_type))
        elif 'CURR' in mode:
            return self.write(':SENS:CURR:AVER:TCON {:s}'.format(filter_type))
        elif 'RES' in mode:
            return self.write(':SENS:RES:AVER:TCON {:s}'.format(filter_type))
        else:
            raise UserWarning('Unknown sense mode')

    def _get_average_state(self):
        mode = self.sense_mode()
        if 'VOLT' in mode:
            return self.ask(':SENS:VOLT:AVER?')
        elif 'CURR' in mode:
            return self.ask(':SENS:CURR:AVER?')
        elif 'RES' in mode:
            return self.ask(':SENS:RES:AVER?')
        else:
            raise UserWarning('Unknown sense mode')

    def _set_average_state(self, value):
        mode = self.sense_mode()
        if 'VOLT' in mode:
            return self.write(':SENS:VOLT:AVER {:d}'.format(value))
        elif 'CURR' in mode:
            return self.write(':SENS:CURR:AVER {:d}'.format(value))
        elif 'RES' in mode:
            return self.write(':SENS:RES:AVER {:d}'.format(value))
        else:
            raise UserWarning('Unknown sense mode')

    def _get_sense_range_auto(self):
        mode = self.sense_mode()
        if 'VOLT' in mode:
            return self.ask(':SENS:VOLT:RANG:AUTO?')
        elif 'CURR' in mode:
            return self.ask(':SENS:CURR:RANG:AUTO?')
        elif 'RES' in mode:
            return self.ask(':SENS:RES:RANG:AUTO?')
        else:
            raise UserWarning('Unknown sense mode')

    def _set_sense_range_auto(self, value):
        mode = self.sense_mode()
        if 'VOLT' in mode:
            return self.write(':SENS:VOLT:RANG:AUTO {:d}'.format(value))
        elif 'CURR' in mode:
            return self.write(':SENS:CURR:RANG:AUTO {:d}'.format(value))
        elif 'RES' in mode:
            return self.write(':SENS:RES:RANG:AUTO {:d}'.format(value))
        else:
            raise UserWarning('Unknown sense mode')

    def _get_sense_range_manual(self):
        mode = self.sense_mode()
        if 'VOLT' in mode:
            return self.ask(':SENS:VOLT:RANG?')+' V'
        elif 'CURR' in mode:
            return self.ask(':SENS:CURR:RANG?')+' A'
        elif 'RES' in mode:
            return self.ask(':SENS:RES:RANG?')+' Ohms'
        else:
            raise UserWarning('Unknown sense mode')

    def _set_sense_range_manual(self, value):
        mode = self.sense_mode()
        if 'VOLT' in mode:
            if value>=0.02 and value<=200.0:
                return self.write(':SENS:VOLT:RANG {:f}'.format(value))
            else:
                raise ValueError('Out of range limits!')
        elif 'CURR' in mode:
            if value>=1e-8 and value<=1.0:
                return self.write(':SENS:CURR:RANG {:f}'.format(value))
            else:
                raise ValueError('Out of range limits!')
        elif 'RES' in mode:
            if value >= 20 and value <= 2e8:
                return self.write(':SENS:RES:RANG {:f}'.format(value))
            else:
                raise ValueError('Out of range limits!')
        else:
            raise UserWarning('Unknown sense mode')

    def _get_sense_range_auto_lower_limit(self):
        mode = self.sense_mode()
        if 'VOLT' in mode:
            return self.ask(':SENS:VOLT:RANG:AUTO:LLIM?')+' V'
        elif 'CURR' in mode:
            return self.ask(':SENS:CURR:RANG:AUTO:LLIM?')+' A'
        elif 'RES' in mode:
            return self.ask(':SENS:RES:RANG:AUTO:LLIM?')+' Ohms'
        else:
            raise UserWarning('Unknown sense mode')

    def _set_sense_range_auto_lower_limit(self, value):
        mode = self.sense_mode()
        if 'VOLT' in mode:
            if value>=0.02 and value<=200.0:
                return self.write(':SENS:VOLT:RANG:AUTO:LLIM {:f}'.format(value))
            else:
                raise ValueError('Out of range limits!')
        elif 'CURR' in mode:
            if value>=1e-8 and value<=1.0:
                return self.write(':SENS:CURR:RANG:AUTO:LLIM {:f}'.format(value))
            else:
                raise ValueError('Out of range limits!')
        elif 'RES' in mode:
            if value >= 2 and value <= 2e8:
                return self.write(':SENS:RES:RANG:AUTO:LLIM {:f}'.format(value))
            else:
                raise ValueError('Out of range limits!')
        else:
            raise UserWarning('Unknown sense mode')

    def _get_sense_range_auto_upper_limit(self):
        mode = self.sense_mode()
        if 'VOLT' in mode:
            return self.ask(':SENS:VOLT:RANG:AUTO:ULIM?')+' V'
        elif 'CURR' in mode:
            raise ValueError('Wrong sense mode for auto range upper limit!')
        elif 'RES' in mode:
            return self.ask(':SENS:RES:RANG:AUTO:ULIM?')+' Ohms'
        else:
            raise UserWarning('Unknown sense mode')

    def _set_sense_range_auto_upper_limit(self, value):
        mode = self.sense_mode()
        if 'VOLT' in mode:
            raise ValueError('Auto range upper limit can be set only for resistance!')
        elif 'CURR' in mode:
            raise ValueError('Auto range upper limit can be set only for resistance!')
        elif 'RES' in mode:
            lower_limit = self.sense_range_auto_lower_limit()
            if value >= 20 and value <= 2e8 and lower_limit <= value:
                return self.write(':SENS:RES:RANG:AUTO:ULIM {:f}'.format(value))
            else:
                raise ValueError('Out of range limits!')
        else:
            raise UserWarning('Unknown sense mode')

    def _get_nplc(self):
        mode = self.sense_mode()
        if 'VOLT' in mode:
            return self.ask(':SENS:VOLT:NPLC?')
        elif 'CURR' in mode:
            return self.ask(':SENS:CURR:NPLC?')
        elif 'RES' in mode:
            return self.ask(':SENS:RES:NPLC?')
        else:
            raise UserWarning('Unknown sense mode')

    def _set_nplc(self, value):
        mode = self.sense_mode()
        if 'VOLT' in mode:
            return self.write(':SENS:VOLT:NPLC {:f}'.format(value))
        elif 'CURR' in mode:
            return self.write(':SENS:CURR:NPLC {:f}'.format(value))
        elif 'RES' in mode:
            return self.write(':SENS:RES:NPLC {:f}'.format(value))
        else:
            raise UserWarning('Unknown sense mode')

    def _get_relative_offset(self):
        mode = self.sense_mode()
        if 'VOLT' in mode:
            return self.ask(':SENS:VOLT:REL?')+' V'
        elif 'CURR' in mode:
            return self.ask(':SENS:CURR:REL?')+' A'
        elif 'RES' in mode:
            return self.ask(':SENS:RES:REL?')+' Ohms'
        else:
            raise UserWarning('Unknown sense mode')

    def _set_relative_offset(self, value):
        mode = self.sense_mode()
        if 'VOLT' in mode:
            if value>=-200.0 and value<=200.0:
                return self.write(':SENS:VOLT:REL {:f}'.format(value))
            else:
                raise ValueError('Out of range limits!')
        elif 'CURR' in mode:
            if value>=-1.0 and value<=1.0:
                return self.write(':SENS:CURR:REL {:f}'.format(value))
            else:
                raise ValueError('Out of range limits!')
        elif 'RES' in mode:
            if value >= -2e8 and value <= 2e8:
                return self.write(':SENS:RES:REL {:f}'.format(value))
            else:
                raise ValueError('Out of range limits!')
        else:
            raise UserWarning('Unknown sense mode')

    def _get_relative_offset_state(self):
        mode = self.sense_mode()
        if 'VOLT' in mode:
            return self.ask(':SENS:VOLT:REL:STAT?')
        elif 'CURR' in mode:
            return self.ask(':SENS:CURR:REL:STAT?')
        elif 'RES' in mode:
            return self.ask(':SENS:RES:REL:STAT?')
        else:
            raise UserWarning('Unknown sense mode')

    def _set_relative_offset_state(self, value):
        mode = self.sense_mode()
        if 'VOLT' in mode:
            return self.write(':SENS:VOLT:REL:STAT {:d}'.format(value))
        elif 'CURR' in mode:
            return self.write(':SENS:CURR:REL:STAT {:d}'.format(value))
        elif 'RES' in mode:
            return self.write(':SENS:RES:REL:STAT {:d}'.format(value))
        else:
            raise UserWarning('Unknown sense mode')

    def _get_four_wire_mode(self):
        mode = self.sense_mode()
        if 'VOLT' in mode:
            return self.ask(':SENS:VOLT:RSEN?')
        elif 'CURR' in mode:
            return self.ask(':SENS:CURR:RSEN?')
        elif 'RES' in mode:
            return self.ask(':SENS:RES:RSEN?')
        else:
            raise UserWarning('Unknown sense mode')

    def _set_four_wire_mode(self, value):
        mode = self.sense_mode()
        if 'VOLT' in mode:
            return self.write(':SENS:VOLT:RSEN {:d}'.format(value))
        elif 'CURR' in mode:
            return self.write(':SENS:CURR:RSEN {:d}'.format(value))
        elif 'RES' in mode:
            return self.write(':SENS:RES:RSEN {:d}'.format(value))
        else:
            raise UserWarning('Unknown sense mode')


    ### Other deprecated functions ###
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
    def _time_parser(self, msg):
        fields = [float(x) for x in msg.split(',')]
        return fields[2]

    # deprecated
    def make_buffer(self, buffer_name, buffer_size):
        self.write('TRACe:MAKE {:s}, {:d}'.format(buffer_name, buffer_size))

    # deprecated
    def clear_buffer(self, buffer_name):
        self.write(':TRACe:CLEar {:s}'.format(buffer_name))

    # deprecated
    # def _source_mode(self):
    #     """
    #     This helper function is used to manage most settable parameters to ensure the device is
    #     consistently configured for the correct output mode.
    #     """
    #     mode = self.source_mode().get_latest()
    #     if mode is not None:
    #         return mode
    #     else:
    #         return self.source_mode()

    # deprecated
    # def _sense_mode(self):
    #     """
    #     This helper function is used to manage most settable parameters to ensure the device is
    #     consistently configured for the correct sensing mode.
    #     """
    #     mode = self.sense_mode().get_latest()
    #     if mode is not None:
    #         return mode
    #     else:
    #         return self.sense_mode()

    # # deprecated
    # def _volt_parser(self, msg):
    #     fields = [float(x) for x in msg.split(',')]
    #     return fields[0]

    # # deprecated
    # def _curr_parser(self, msg):
    #     fields = [float(x) for x in msg.split(',')]
    #     return fields[1]

    # # deprecated
    # def _resistance_parser(self, msg):
    #     fields = [float(x) for x in msg.split(',')]
    #     return fields[0]/fields[1]

    # deprecated
    # def measFunc(self):
    #     self.write('OUTput ON')
    #     self.write('TRACe:TRIGger "MykhBuffer1"')
    #     return self.ask('TRACe:DATA? 1, 1, "MykhBuffer1", SOUR, READ, SEC')

    # deprecated
    # def measPosFunc(self):
    #     self.write('SOURce:CURR 0.02')
    #     self.write('OUTput ON')
    #     self.write('TRACe:TRIGger "MykhBuffer1"')
    #     return self.ask('TRACe:DATA? 1, 1, "MykhBuffer1", SOUR, READ, SEC')

    # deprecated
    # def measNegFunc(self):
    #     self.write('SOURce:CURR -0.02')
    #     self.write('OUTput ON')
    #     self.write('TRACe:TRIGger "MykhBuffer1"')
    #     return self.ask('TRACe:DATA? 1, 1, "MykhBuffer1", SOUR, READ, SEC')

    # deprecated
    # def getTime(self):
    #     return self.ask('TRACe:DATA? 1, 1, "MykhBuffer1", SOUR, READ, SEC')

    # deprecated
    # def getCurrent(self):
    #     return self.ask('TRACe:DATA? 1, 1, "MykhBuffer1", SOUR, READ, SEC')

    # deprecated
    # def setVoltSens(self):
    #     self.write('*RST')
    #     self.write(':ROUT:TERM REAR')
    #     self.write('SENSe:FUNCtion "VOLT"')
    #     self.write('SENSe:VOLTage:RANGe:AUTO ON')
    #     self.write('SENSe:VOLTage:UNIT VOLT')
    #     self.write('SENSe:VOLTage:RSENse ON')
    #     self.write('SOURce:FUNCtion CURR')
    #     self.write('SOURce:CURR 0.02')
    #     self.write('SOURce:CURR:VLIM 2')
    #     self.write('SENSe:COUNT 1')
    #     self.write(':SENSe:VOLTage:NPLCycles 10')
    #     self.write(':DISPlay:VOLTage:DIGits 6')