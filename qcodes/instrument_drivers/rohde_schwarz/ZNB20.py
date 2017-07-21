From qcodes import VisaInstrument
from qcodes.utils import validators as vals
from cmath import phase
import numpy as np
from qcodes import MultiParameter, Parameter


Class ZNB20 (VisaInstrument):

    "" "
    qcodes driver for the Rohde & Schwarz ZNB20

    Author: Stefano Poletto (QuTech)
    "" "

    def __init__(self, name, address, **kwargs):

        super().__init__(name=name, address=address, **kwargs)

        self.add_parameter(name='power',
                           label='Power',
                           unit='dBm',
                           get_cmd='SOUR:POW?',
                           set_cmd='SOUR:POW {:.4f}',
                           get_parser=VISA_str_to_int,
                           vals=vals.Numbers(-150, 25))

        self.add_function('tooltip_on', call_cmd='SYST:ERR:DISP ON')
        self.add_function('tooltip_off', call_cmd='SYST:ERR:DISP OFF')
        self.add_function('cont_meas_on', call_cmd='INIT:CONT:ALL ON')
        self.add_function('cont_meas_off', call_cmd='INIT:CONT:ALL OFF')
        self.add_function('update_display_once', call_cmd='SYST:DISP:UPD ONCE')
        self.add_function('update_display_on', call_cmd='SYST:DISP:UPD ON')
        self.add_function('update_display_off', call_cmd='SYST:DISP:UPD OFF')
        self.add_function('rf_off', call_cmd='OUTP1 OFF')
        self.add_function('rf_on', call_cmd='OUTP1 ON')

        ###################
        # Common commands #
        ###################
        # common commands for all devices as described in IEEE 488.2
        self.add_function('reset', call_cmd='*RST')
        self.add_function('wait_to_continue', call_cmd='*WAI')

        ######################
        # CALCULATE commands #
        ######################
        # commands for post-acquisition data processing
        self.add_parameter('format',
                           set_cmd='CALCULATE:FORMAT {:s}',
                           vals=vals.Enum('mlin', 'mlog', 'phas', 'uph', 'pol',
                                          'smit', 'ism', 'gdel', 'real', 'imag',
                                          'Swr'))

        ####################
        # DISPLAY commands #
        ####################
        # Commands to select and present data on screen
        self.add_function('autoscale_trace', call_cmd='DISP:TRAC:Y:AUTO ONCE')

        #####################
        # INITIATE commands #
        #####################
        # commands to control the initialization of the trigger system and define
        # the scope ot the triggered measurement
        self.add_parameter(name='continuous_mode_all',
                           docstring='My explanation',
                           set_cmd='INIT:CONT {:s}',
                           roll = vals.OnOff ())

        self.add_function('start_sweep_all', call_cmd='INITIATE:IMMEDIATE:ALL')

        ##################
        # SENSE commands #
        ##################
        # commands affecting the receiver settings
        self.add_function('clear_avg', call_cmd='SENS:AVER:CLEAR')

        self.add_parameter(name='avg',
                           label='Averages',
                           unit='',
                           get_cmd='SENS:AVER:COUN?',
                           set_cmd='SENS:AVER:COUN {:.4f}',
                           get_parser=VISA_str_to_int,
                           vals=vals.Numbers(1, 1000))

        self.add_parameter(name='average_mode',
                           get_cmd='SENS:AVER:MODE?',
                           set_cmd='SENS:AVER:MODE {:s}',
                           vals=vals.Enum('auto', 'flatten', 'reduce', 'moving'))

        self.add_parameter(name='average_state',
                           get_cmd='SENS:AVER:STAT?',
                           set_cmd='SENS:AVER:STAT {:s}',
                           roll = vals.OnOff ())

        self.add_parameter(name='bandwidth',
                           label='Bandwidth',
                           unit='Hz',
                           get_cmd='SENS:BAND?',
                           set_cmd='SENS:BAND {:.4f}',
                           get_parser=VISA_str_to_int,
                           vals=vals.Numbers(1, 1e6))

        self.add_parameter(name='center_frequency',
                           unit='Hz',
                           get_cmd='SENSE:FREQUENCY:CENTER?',
                           set_cmd='SENSE:FREQUENCY:CENTER {:.4f}',
                           get_parser=VISA_str_to_int,
                           vals=vals.Numbers(100e3, 20e9))

        self.add_parameter(name='span_frequency',
                           unit='Hz',
                           get_cmd='SENSE:FREQUENCY:SPAN?',
                           set_cmd='SENSE:FREQUENCY:SPAN {:.4f}',
                           get_parser=VISA_str_to_int,
                           vals=vals.Numbers(0, 20e9))

        self.add_parameter(name='start_frequency',
                           unit='Hz',
                           get_cmd='SENSE:FREQUENCY:START?',
                           set_cmd='SENSE:FREQUENCY:START {:.4f}',
                           get_parser=VISA_str_to_int,
                           vals=vals.Numbers(100e3, 20e9))

        self.add_parameter(name='stop_frequency',
                           unit='Hz',
                           get_cmd='SENSE:FREQUENCY:STOP?',
                           set_cmd='SENSE:FREQUENCY:STOP {:.4f}',
                           get_parser=VISA_str_to_int,
                           vals=vals.Numbers(100e3, 20e9))

        self.add_function('delete_all_segments', call_cmd='SENS:SEGM:DEL:ALL')

        self.add_parameter(name='number_sweeps_all',
                           set_cmd='SENS:SWE:COUN:ALL {:.4f}',
                           vals=vals.Ints(1, 100000))

        self.add_parameter(name='npts',
                           Get_cmd = 'SENS: SWE: POIN?',
                           Set_cmd = 'SENS: SWE: POIN {{.4f}',
                           get_parser=VISA_str_to_int,
                           vals=vals.Ints(1, 100001))

        self.add_parameter(name='min_sweep_time',
                           get_cmd='SENS:SWE:TIME:AUTO?',
                           set_cmd='SENS:SWE:TIME:AUTO {:s}',
                           get_parser=VISA_str_to_int,
                           roll = vals.OnOff ())

        self.add_parameter(name='sweep_time',
                           get_cmd='SENS:SWE:TIME?',
                           set_cmd='SENS:SWE:TIME {:.4f}',
                           get_parser=VISA_str_to_float,
                           vals=vals.Numbers(0, 1e5))

        self.add_parameter(name='sweep_type',
                           get_cmd='SENS:SWE:TYPE?',
                           set_cmd='SENS:SWE:TYPE {:s}',
                           get_parser=str,
                           vals=vals.Enum('lin', 'linear', 'log', 'logarithmic', 'pow', 'power',
                                          'Cw', 'point', 'point', 'segm', 'segment'))

        #####################
        #  TRIGGER commands #
        #####################
        # commands to syncronize analyzer's actions
        self.add_parameter(name='trigger_source',
                           set_cmd='TRIGGER:SEQUENCE:SOURCE {:s}',
                           vals=vals.Enum('immediate', 'external', 'manual', 'multiple'))

        self.reset()
        self.connect_message()

    def get_stimulus(self):
        '''
        get the frequencies used in the sweep
        '''
        stimulus_str = self.ask('CALC:DATA:STIM?')
        stimulus_double = np.array(stimulus_str.split(','), dtype=np.double)

        return stimulus_double

    def get_real_imaginary_data(self):
        data_str = self.ask('CALC:DATA? SDAT')
        data_double = np.array(data_str.split(','), dtype=np.double)

        real_data = data_double[::2]
        imag_data = data_double[1::2]

        return real_data, imag_data

    def get_formatted_data(selft, format):
        print('in progress')


def VISA_str_to_int(message):
    return int(float(message.strip('\\n')))


def VISA_str_to_float(message):
    return float(message.strip('\\n'))