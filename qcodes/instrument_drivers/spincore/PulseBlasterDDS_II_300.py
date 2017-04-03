from qcodes.instrument.base import Instrument
from qcodes.utils.validators import Numbers, Enum, Ints, Strings, Anything
from functools import partial
try:
    from .spinapi import *
except:
    raise ImportError('To use a SpinCore PulseBlaster, install the Python SpinAPI')

class PB_KEYWORDS():
    ns = 1.0
    us = 1000.0
    ms = 1000000.0

    MHz = 1.0
    kHz = 0.001
    Hz  = 0.000001

    # pb_start_programming options
    PULSE_PROGRAM  = 0
    FREQ_REGS      = 1
    TX_PHASE_REGS  = 2
    RX_PHASE_REGS  = 3
    #COS_PHASE_REGS = 4 # RadioProcessor boards ONLY
    #SIN_PHASE_REGS = 5 # RadioProcessor boards ONLY

class PB_ERRORS():
    pass


class PB_DDS(Instrument):
    """
    This is the qcodes driver for the SpinCore PulseBlasterDDS-II-300

    Args:
        name (str): name for this instrument, passed to the base instrument
    """

    def __init__(self, name, **kwargs):
        if 'Werror' in kwargs:
            self.werror = kwargs.pop('Werror')
        else:   
            self.werror = False

        super().__init__(name, **kwargs)

        self.N_CHANNELS        = 2
        self.N_FREQ_REGS       = 16
        self.N_PHASE_REGS      = 8
        self.N_AMPLITUDE_REGS  = 4

        
        self.DEFAULT_DDS_INST      = (0, 0, 0, 0, 0)
            
        # Initialise the DDS
        # NOTE: Hard coded value for board may want to be part of instrument init
        if self.count_boards() > 0:
            self.select_board(0)
            pb_init()
            # Call set defaults as part of init to give board a well defined state
            pb_set_defaults()
        else:
            if self.werror:
                raise IOError('PB Error: Can\'t find board')
            else:
                print('Error: Can\'t find board, will continue anyway')

        # Internal State Variables, used when setting individual registers when others
        # may be undefined
        self.__frequency = [[0.0]*self.N_FREQ_REGS     , [0.0]*self.N_FREQ_REGS ]
        self.__phase     = [[0.0]*self.N_PHASE_REGS    , [0.0]*self.N_PHASE_REGS]
        # Create an empty list of lists of instructions [[], [], ...]
        self.__chan_instructions = []
        for n in range(self.N_CHANNELS):
            self.__chan_instructions.append([])

        ###########################################################################
        ###                              Parameters                             ###
        ###########################################################################
        for n in range(self.N_CHANNELS):
            # DDS Register Bank
            for r in range(self.N_FREQ_REGS):
                self.add_parameter(
                    'frequency_n{}_r{}'.format(n, r),
                    label='DDS Frequency for channel {}, register {}'.format(n, r),
                    set_cmd=partial(self.set_frequency_register, channel=n, register=r),
                    vals=Numbers(),
                    docstring=''
                )

            for r in range(self.N_PHASE_REGS):
                self.add_parameter(
                    'phase_n{}_r{}'.format(n, r),
                    label='DDS Phase for channel {}, register {}'.format(n, r),
                    set_cmd=partial(self.set_phase_register, channel=n, register=r),
                    vals=Numbers(),
                    docstring=''
                )

            for r in range(self.N_AMPLITUDE_REGS):
                self.add_parameter(
                    'amplitude_n{}_r{}'.format(n, r),
                    label='DDS Amplitude for channel {}, register {}'.format(n, r),
                    set_cmd=partial(self.set_amplitude_register, channel=n, register=r),
                    vals=Numbers(),
                    docstring=''
                )

    ###########################################################################
    ###                         DDS Board commands                          ###
    ###########################################################################
    # Just wrapped from spinapi.py #
    
    def get_version(self):
        ret = pb_get_version()
        if not isinstance(ret, str):
            raise IOError(self.get_error())
        return ret

    def get_error(self):
        """ Print library error as UTF-8 encoded string."""
        ret = "PB Error: " + pb_get_error()
        
    def count_boards(self):
        """ Print the number of boards detected in the system."""
        ret = pb_count_boards()
        if (ret < 0):
            raise IOError(self.get_error())
        return ret
        
    def init(self):
        """ Initialize currently selected board."""
        ret = pb_init()
        if (ret < 0):
            raise IOError(self.get_error())
        return ret
        
    def set_debug(self, debug):
        ret = pb_set_debug(debug)
        if (ret < 0):
            raise IOError(self.get_error())
        return ret
        
    def select_board(self, board_number):
        """ Select a specific board number"""
        ret = pb_select_board(board_number)
        if (ret < 0):
            raise IOError(self.get_error())
        return ret
        
    def set_defaults(self):
        """ Set board defaults. Must be called before using any other board functions."""
        ret = pb_set_defaults()
        if (ret < 0):
            raise IOError(self.get_error())
        return ret
        
    def _core_clock(self, clock):
        ret = pb_core_clock(ctypes.c_double(clock))
        if (ret < 0):
            raise IOError(self.get_error())
        return ret
        
    def write_register(self, address, value):
        ret = pb_write_register(address, value)
        if (ret < 0):
            raise IOError(self.get_error())
        return ret

    ###########################################################################
    ###                        DDS Control commands                         ###
    ###########################################################################

    def add_inst(self, inst, channel):
        self.__chan_instructions[channel].append(inst)

    def get_inst_list(self, channel):
        return self.__chan_instructions[channel]

    def program_pulse_sequence(self, pulse_sequence):
        """ An all-in-one function to send a pulse sequence to the board

        Args:
            pulse_sequence (list) : a list of instructions to program the board
                                    the instructions should be tuples i.e.
                                    (FREQ0, PHASE0, ...)
        """
        pb_start_programming(PB_KEYWORDS.PULSE_PROGRAM)
        for p in pulse_sequence:
            pb_inst_dds2(*p)
        pb_stop_programming()
    

    def program_inst(self):
        """ Send the current instruction list to the board
        
        Args:
            
        """
        raise IOError('Warning: Unimplemented function')
        pass
        #pb_start_programming()
        #n_inst = 0
        ## Find the longest set of instructions defined
        #for n in range(self.N_CHANNELS):
        #    n_inst = max(n_inst, len(self.__chan_instructions[n]))
        #for i in range(n_inst):
        #    # TODO: use lambdas here for ease
        #    #if (self.__chan_instructions[
        #    pass
        #pb_stop_programming()

    def set_shape_defaults(self, channel):
        """ Resets the shape for the specified channel
        
        Args:
            channel      (int) : Either DDS0 (0) or DDS1 (1)
        """
        pb_select_dds(channel)
        ret = pb_set_shape_defaults()
        if (ret < 0):
            raise IOError(self.get_error())
        return ret

    def load_waveform(self, data, device, channel):
        """ Loads a waveform onto the DDS

        Args:
            data       (float) : an array of 1024 points representing a single 
                                 period of the waveform. Points range between 
                                 -1.0 and 1.0
            device       (int) : Either DEVICE_DDS or DEVICE_SHAPE
            channel      (int) : Either DDS0 (0) or DDS1 (1)
        """
        pb_select_dds(channel)
        ret = pb_dds_load(data, device)
        if (ret < 0):
            raise IOError(self.get_error())
        return ret

    def set_envelope_freq(self, freq, register, channel):
        """ Sets the frequency for an envelope register
        
        Args:
            freq       (float) : the frequency in MHz for the envelope register
            register     (int) : the register number
            channel      (int) : Either DDS0 (0) or DDS1 (1)
        """
        pb_select_dds(channel)
        ret = pb_dds_set_envelope_freq(freq, register)
        if (ret < 0):
            raise IOError(self.get_error())
        return ret

    ###########################################################################
    ###                  Set/Get commands for parameters                    ###
    ###########################################################################

    def set_frequency_register(self, frequency, channel, register):
        """ Sets the DDS frequency for the specified channel and register
        
        Args:
            frequency (double) : the frequency in Mhz to write to the register
            register     (int) : the register number
            channel      (int) : Either DDS0 (0) or DDS1 (1)
        """
        self.__frequency[channel][register] = frequency
        pb_select_dds(channel)
        pb_start_programming(PB_KEYWORDS.FREQ_REGS)
        for r in range(self.N_FREQ_REGS):
            ret = pb_set_freq(self.__frequency[channel][r])
            if (ret < 0):
                raise IOError(self.get_error())
        ret = pb_stop_programming()
        if (ret < 0):
            raise IOError(self.get_error())
        return ret

    def set_phase_register(self, phase, channel, register):
        """ Sets the DDS phase for the specified channel and register
        
        Args:
            phase     (double) : the phase in degrees to write to the register
            register     (int) : the register number
            channel      (int) : Either DDS0 (0) or DDS1 (1)
        """
        self.__phase[channel][register] = phase
        pb_select_dds(channel)
        pb_start_programming(PB_KEYWORDS.TX_PHASE_REGS)
        for r in range(self.N_PHASE_REGS):
            pb_set_phase(self.__phase[channel][register])
        ret = pb_stop_programming()
        if (ret < 0):
            raise IOError(self.get_error())
        return ret

    def set_amplitude_register(self, amplitude, channel, register):
        """ Sets the DDS amplitude for the specified channel and register
        
        Args:
            amplitude (double) : the amplitude in volts to write to the register
            register     (int) : the register number
            channel      (int) : Either DDS0 (0) or DDS1 (1)
        """
        pb_select_dds(channel)
        ret = self.pb_set_amp(amplitude, register)
        if (ret < 0):
            raise IOError(self.get_error())
        return ret

