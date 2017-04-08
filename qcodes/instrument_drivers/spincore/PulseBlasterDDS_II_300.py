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
    s  = 1000000000.0

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

    # pb_write_register options
    BASE_ADDR       = 0x40000
    FLAG_STATES     = BASE_ADDR + 0x8
    START_LOCATION  = BASE_ADDR + 0x7

    # pb_dds_load options
    DEVICE_DDS   = 0
    DEVICE_SHAPE = 1



class PB_DDS(Instrument):
    """
    This is the qcodes driver for the SpinCore PulseBlasterDDS-II-300

    Args:
        name (str): name for this instrument, passed to the base instrument
    """
    def error_parse(self, value):
        if not isinstance(value, str) and value < 0:
            raise IOError(self.get_error())
        return value

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
            self.init()
            self.select_board(0)
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

        self.add_parameter(
            'core_clock',
            label='The core clock of the PulseBlasterDDS',
            set_cmd=self.set_core_clock,
            vals=Numbers(),
            docstring=''
        )

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
        """ Return the current version of the spinapi library being used. """
        return self.error_parse(pb_get_version())

    def get_error(self):
        """ Print library error as UTF-8 encoded string. """
        ret = "PB Error: " + pb_get_error()
        
    def count_boards(self):
        """ Print the number of boards detected in the system. """
        return self.error_parse(pb_count_boards())
        
    def init(self):
        """ Initialize currently selected board. """
        return self.error_parse(pb_init())
        
    def set_debug(self, debug):
        """ Enables logging to a log.txt file in the current directory """
        return self.error_parse(pb_set_debug(debug))
        
    def select_board(self, board_number):
        """ Select a specific board number """
        return self.error_parse(pb_select_board(board_number))
        
    def set_defaults(self):
        """ Set board defaults. Must be called before using any other board functions."""
        return self.error_parse(pb_set_defaults())
        
    def set_core_clock(self, clock):
        """ Sets the core clock reference value
        
        Note: This does not change anything in the device, it just provides the library
              with the value given
        """
        pb_core_clock(clock)
        
    def write_register(self, address, value):
        """ Write to one of two core registers in the DDS
        
        Args:
            address (int) : the address of the register
            value   (int) : the value to be written to the register
        """
        return self.error_parse(pb_write_register(address, value))

    ###########################################################################
    ###                        DDS Control commands                         ###
    ###########################################################################

    def start(self):
        return self.error_parse(pb_start())

    def reset(self):
        return self.error_parse(pb_reset())

    def stop(self):
        return self.error_parse(pb_stop())

    def close(self):
        return self.error_parse(pb_close())

    def add_inst(self, inst, channel):
        self.__chan_instructions[channel].append(inst)

    def get_inst_list(self, channel):
        return self.__chan_instructions[channel]

    def start_programming(self, mode):
        """ Start a programming sequence 
        
        Args:
            mode    (int) : one of (PULSE_PROGRAM, FREQ_REGS, etc.)
        """
        return self.error_parse(pb_start_programming(mode))

    def inst_dds2(self, inst):
        """ During start_programming(PULSE_PROGRAM) 
            add an unshaped sine pulse to program
        
        Args:
            inst  (tuple) : a tuple to program the board in the
                            (FREQ0, PHASE0, ...)
        """
        return self.error_parse(pb_inst_dds2(*inst))

    def inst_dds2_shape(self, inst):
        """ During start_programming(PULSE_PROGRAM) 
            add a shaped pulse to program, if desired
        
        Args:
            inst  (tuple) : a tuple to program the board in the
                            (FREQ0, PHASE0, ...)
        """
        return self.error_parse(pb_inst_dds2_shape(*inst))

    def dds_load(self, data, device):
        """ Load a 1024 point waveform into one of two devices

        Args:
            data  (list) : a 1024 point waveform. Each point ranges
                           from -1.0 to 1.0
            device (int) : Either DEVICE_DDS (the default shape for all output)
                               or DEVICE_SHAPE (an alternative shape register)
        """
        return self.error_parse(pb_dds_load())

    def stop_programming(self):
        """ End a programming sequence """
        return self.error_parse(pb_stop_programming())


    def program_pulse_sequence(self, pulse_sequence):
        """ An all-in-one function to send a pulse sequence to the board

        Args:
            pulse_sequence (list) : a list of instructions to program the board
                                    the instructions should be tuples i.e.
                                    (FREQ0, PHASE0, ...)
        """
        pb_start_programming(PB_KEYWORDS.PULSE_PROGRAM)
        for p in pulse_sequence:
            # convert last element from seconds to ns
            p = p[:-1] + (p[-1]*PB_KEYWORDS.s,)
            # * breaks tuple into args, 
            pb_inst_dds2(*p)
        pb_stop_programming()
    

    def program_inst(self):
        """ Send the current instruction list to the board
        
        Args:
            
        """
        raise NotImplementedError('Warning: Unimplemented function')
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
        return self.error_parse(pb_set_shape_defaults())

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
        return self.error_parse(pb_dds_load(data, device))

    def set_envelope_freq(self, freq, register, channel):
        """ Sets the frequency for an envelope register
        
        Args:
            freq       (float) : the frequency in Hz for the envelope register
            register     (int) : the register number
            channel      (int) : Either DDS0 (0) or DDS1 (1)
        """
        # Scale the frequency to Hertz, as the underlying api assumes MHz
        freq *= PB_KEYWORDS.Hz 
        pb_select_dds(channel)
        return self.error_parse(pb_dds_set_envelope_freq(freq, register))

    ###########################################################################
    ###                  Set/Get commands for parameters                    ###
    ###########################################################################

    def set_frequency_register(self, frequency, channel, register):
        """ Sets the DDS frequency for the specified channel and register
        
        Args:
            frequency (double) : the frequency in Hz to write to the register
            register     (int) : the register number
            channel      (int) : Either DDS0 (0) or DDS1 (1)
        """
        # Scale the frequency to Hertz, as the underlying api assumes MHz
        frequency *= PB_KEYWORDS.Hz 
        self.__frequency[channel][register] = frequency
        pb_select_dds(channel)
        self.start_programming(PB_KEYWORDS.FREQ_REGS)
        for r in range(self.N_FREQ_REGS):
            self.error_parse(pb_set_freq(self.__frequency[channel][r]))
        self.stop_programming()

    def set_phase_register(self, phase, channel, register):
        """ Sets the DDS phase for the specified channel and register
        
        Args:
            phase     (double) : the phase in degrees to write to the register
            register     (int) : the register number
            channel      (int) : Either DDS0 (0) or DDS1 (1)
        """
        self.__phase[channel][register] = phase
        pb_select_dds(channel)
        self.start_programming(PB_KEYWORDS.TX_PHASE_REGS)
        for r in range(self.N_PHASE_REGS):
            self.error_parse(pb_set_phase(self.__phase[channel][register]))
        self.stop_programming()

    def set_amplitude_register(self, amplitude, channel, register):
        """ Sets the DDS amplitude for the specified channel and register
        
        Args:
            amplitude (double) : the amplitude in volts to write to the register
            register     (int) : the register number
            channel      (int) : Either DDS0 (0) or DDS1 (1)
        """
        pb_select_dds(channel)
        return self.error_parse(pb_set_amp(amplitude, register))

