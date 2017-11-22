from functools import partial, wraps
import logging
import numpy as np

from qcodes import Instrument
from qcodes.utils.validators import Lists, Numbers
from qcodes.instrument.channel import InstrumentChannel, ChannelList

try:
    from . import spinapi as api
except:
    raise ImportError(
        'To use a SpinCore PulseBlaster, install the Python SpinAPI')

logger = logging.getLogger(__name__)

# Add seconds to list of PulseBlaster keywords
api.s = 1# TODO fix 1000000000.0


def error_parse(f):
    """ Decorator for DLL communication that checks for errors"""

    @wraps(f)
    def error_wrapper(*args, **kwargs):
        value = f(*args, **kwargs)
        if not isinstance(value, str) and value < 0:
            logger.error('{}: {}'.format(value, api.pb_get_error()))
            raise IOError('{}: {}'.format(value, api.pb_get_error()))
        return value

    return error_wrapper


class PulseBlasterDDS(Instrument):
    """
    This is the qcodes driver for the SpinCore PulseBlasterDDS-II-300
    """

    # pb_start_programming options
    TX_PHASE_REGS = 2
    RX_PHASE_REGS = 3
    # COS_PHASE_REGS = 4 # RadioProcessor boards ONLY
    # SIN_PHASE_REGS = 5 # RadioProcessor boards ONLY

    # pb_write_register options
    BASE_ADDR = 0x40000
    FLAG_STATES = BASE_ADDR + 0x8
    START_LOCATION = BASE_ADDR + 0x7

    # pb_dds_load options
    DEVICE_DDS = 0x099000
    DEVICE_SHAPE = 0x099001

    N_CHANNELS = 2
    N_FREQ_REGS = 16
    N_PHASE_REGS = 8
    N_AMPLITUDE_REGS = 4

    DEFAULT_DDS_INST = (0, 0, 0, 0, 0)

    PROGRAM_INSTRUCTIONS_MAP = {
        'CONTINUE': 0,  #inst_data=Not used
        'STOP': 1,      #inst_data=Not used
        'LOOP': 2,      #inst_data=Number of desired loops
        'END_LOOP': 3,  #inst_data=Address of instruction originating loop
        'JSR': 4,       #inst_data=Address of first instruction in subroutine
        'RTS': 5,       #inst_data=Not Used
        'BRANCH': 6,    #inst_data=Address of instruction to branch to
        'LONG_DELAY': 7,#inst_data=Number of desired repetitions
        'WAIT': 8}      #inst_data=Not used

    def __init__(self, name, board_number=0, initialize=True, **kwargs):
        """
        Initialize the pulseblaster DDS

        Args:
            name: Name of instrument 
            **kwargs: Additional keyword arguments passed to Instrument
        """
        super().__init__(name, **kwargs)

        # Create an empty list of lists of instructions [[], [], ...]
        self.instructions = [[] for _ in range(self.N_CHANNELS)]


        ########################################################################
        ###                              Parameters                          ###
        ########################################################################
        self.add_parameter(
            'core_clock',
            label='Core clock',
            set_cmd=self.set_core_clock,
            vals=Numbers(),
            docstring='The core clock of the PulseBlasterDDS')

        self.add_parameter('board_number',
                           set_cmd=None,
                           initial_value=board_number)

        self.output_channels = ChannelList(self,
                                           name='output_channels',
                                           chan_type=InstrumentChannel)

        for ch_idx in range(self.N_CHANNELS):
            ch_name = f'ch{ch_idx+1}'
            output_channel = InstrumentChannel(self, ch_name)
            output_channel.idx = ch_idx
            self.output_channels.append(output_channel)

            output_channel.add_parameter(
                'frequencies',
                label=f'{ch_name} frequency',
                unit='Hz',
                # set_cmd=partial(self.set_frequencies, ch_idx),
                set_cmd=None,
                vals=Lists(Numbers()))
            output_channel.add_parameter(
                'phases',
                label=f'{ch_name} phase',
                unit='deg',
                # set_cmd=partial(self.set_phases, ch_idx),
                set_cmd=None,
                vals=Lists(Numbers()))
            output_channel.add_parameter(
                'amplitudes',
                label=f'{ch_name} amplitude',
                unit='V',
                # set_cmd=partial(self.set_amplitudes, ch_idx),
                set_cmd=None,
                vals=Lists(Numbers()))


        self.add_parameter('instruction_sequence',
                           set_cmd=None,
                           initial_value=[],
                           vals=Lists())

        # Initialize the DDS
        self.setup(initialize=initialize)

    ###########################################################################
    ###                         DDS Board commands                          ###
    ###########################################################################
    # Just wrapped from spinapi.py #

    @staticmethod
    def get_error():
        """ Print library error as UTF-8 encoded string. """
        return str(api.pb_get_error())

    @staticmethod
    @error_parse
    def get_version():
        """ Return the current version of the spinapi library being used. """
        return api.pb_get_version()

    @staticmethod
    @error_parse
    def count_boards():
        """ Print the number of boards detected in the system. """
        return api.pb_count_boards()

    @staticmethod
    @error_parse
    def initialize():
        """Initialize currently selected board."""
        return api.pb_init()

    @staticmethod
    @error_parse
    def set_debug(debug):
        """ Enables logging to a log.txt file in the current directory """
        return api.pb_set_debug(debug)

    @staticmethod
    @error_parse
    def select_board(board_number):
        """ Select a specific board number """
        return api.pb_select_board(board_number)

    @staticmethod
    @error_parse
    def set_defaults():
        """ Set board defaults. Must be called before using any other board 
        functions."""
        return api.pb_set_defaults()

    def set_core_clock(self, clock):
        """ Sets the core clock reference value

        Note: This does not change anything in the device, it just provides 
        the library with the value given
        """
        self.select_board(self.board_number())
        api.pb_core_clock(clock)

    @staticmethod
    @error_parse
    def write_register(address, value):
        """ Write to one of two core registers in the DDS

        Args:
            address (int) : the address of the register
            value   (int) : the value to be written to the register
        """
        return api.pb_write_register(address, value)

    ###########################################################################
    ###                        DDS Control commands                         ###
    ###########################################################################


    def setup(self, initialize=False):
        """
        Sets up the DDS
        Args:
            initialize (Bool): Initialize the DDS (should only be done once 
                at the start). False by default 

        """
        assert self.count_boards() > 0, "PB Error: Can't find board"
        self.select_board(self.board_number())
        if initialize:
            self.initialize()

    @error_parse
    def start(self):
        self.setup(initialize=False)

        # Reprogram instruction sequence as it may have been overwritten by stop
        self.program_instruction_sequence(self.instruction_sequence())
        self.reset()

        # This api function does not seem to do much
        return api.pb_start()

    @error_parse
    def reset(self):
        self.select_board(self.board_number())
        return api.pb_reset()

    @error_parse
    def stop(self):
        self.setup(initialize=False)

        # Must overwrite the sequence (api.pb_stop does not work)
        stop_instruction = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'stop', 0, 20000)
        self.program_instruction_sequence([stop_instruction])
        self.reset()

        # This api function does not seem to do much, might be due to
        # firmware (see manual)
        return api.pb_stop()

    @error_parse
    def close(self):
        self.select_board(self.board_number())
        return api.pb_close()

    def add_instruction(self, inst, channel):
        self.instructions[channel].append(inst)

    @error_parse
    def start_programming(self, mode):
        """ Start a programming sequence 

        Args:
            mode    (int) : one of (PULSE_PROGRAM, FREQ_REGS, etc.)
        """
        self.select_board(self.board_number())
        return api.pb_start_programming(mode)

    @error_parse
    def instruction_sine_pulse(self, freq0, phase0, amp0, dds_en0, phase_reset0,
                               freq1, phase1, amp1, dds_en1, phase_reset1,
                               flags, inst, inst_data, length):
        """ During start_programming(PULSE_PROGRAM) 
            add an unshaped sine pulse to program

        Args:
            inst  (tuple) : a tuple to program the board, must be of form:
            
                            (int freq0, int phase0, int amp0, int dds_en0, 
                             int phase_reset0, int freq1, int phase1, 
                             int amp1, int dds_en1, int phase_reset1, int flags,
                             int inst, int inst_data, double length)
                             
                             dds_en should be api.TX_ENABLE or api.TX_DISABLE 
        """
        length = round(length * api.s)

        if isinstance(inst, str):
            inst = self.PROGRAM_INSTRUCTIONS_MAP[inst.upper()]

        self.select_board(self.board_number())
        return api.pb_inst_dds2(freq0, phase0, amp0, dds_en0, phase_reset0,
                                freq1, phase1, amp1, dds_en1, phase_reset1,
                                flags, inst, inst_data, length)

    @error_parse
    def instruction_shaped_sine_pulse(self, *inst):
        """ During start_programming(PULSE_PROGRAM) 
            add a shaped pulse to program, if desired

        Args:
            inst  (tuple) : a tuple to program the board in the
                            (FREQ0, PHASE0, ...)
        """
        inst = inst[:-1] + (round(inst[-1] * api.s),)
        self.select_board(self.board_number())
        return api.pb_inst_dds2_shape(*inst)

    @error_parse
    def dds_load(self, data, device):
        """ Load a 1024 point waveform into one of two devices

        Args:
            data  (list) : a 1024 point waveform. Each point ranges
                           from -1.0 to 1.0
            device (int) : Either DEVICE_DDS (the default shape for all output)
                               or DEVICE_SHAPE (an alternative shape register)
        """
        self.select_board(self.board_number())
        return api.pb_dds_load(device)

    @error_parse
    def stop_programming(self):
        """ End a programming sequence """
        self.select_board(self.board_number())
        return api.pb_stop_programming()

    @error_parse
    def select_dds(self, channel):
        self.select_board(self.board_number())
        return api.pb_select_dds(channel)

    def program_instruction_sequence(self, instruction_sequence):
        """ An all-in-one method to send a sequence of instructions to the board

        Args:
            instruction_sequence (list) : a list of instructions to program
                                    the board
                                    the instructions should be tuples i.e.
                                    (FREQ0, PHASE0, ...)
        """
        self.start_programming(api.PULSE_PROGRAM)
        for instruction in instruction_sequence:
            self.instruction_sine_pulse(*instruction)
        self.stop_programming()

    @error_parse
    def set_shape_defaults(self, channel):
        """ Resets the shape for the specified channel

        Args:
            channel      (int) : Either DDS0 (0) or DDS1 (1)
        """
        self.select_dds(channel)
        return api.pb_set_shape_defaults()

    def load_waveform(self, data, device, channel):
        """ Loads a waveform onto the DDS

        Args:
            data         (arr) : an array of 1024 points representing a single 
                                 period of the waveform. Points range between 
                                 -1.0 and 1.0
            device       (int) : Either DEVICE_DDS or DEVICE_SHAPE
            channel      (int) : Either DDS0 (0) or DDS1 (1)
        """
        self.select_dds(channel)
        return self.dds_load(data, device)

    @error_parse
    def set_envelope_freq(self, freq, channel, register):
        """ Sets the frequency for an envelope register

        Args:
            freq       (float) : the frequency in Hz for the envelope register
            register     (int) : the register number
            channel      (int) : Either DDS0 (0) or DDS1 (1)
        """
        # Scale the frequency to Hertz, as the underlying api assumes MHz
        freq *= api.Hz
        self.select_dds(channel)
        return api.pb_dds_set_envelope_freq(freq, register)

    ###########################################################################
    ###                  Set/Get commands for parameters                    ###
    ###########################################################################

    def set_frequencies(self, channel, frequencies):
        """ Sets the DDS frequency for the specified channel and register

        Args:
            frequency (list(double)): the frequency in Hz for a channel register
            channel      (int) : Either DDS0 (0) or DDS1 (1)
        """
        # Scale the frequency to Hertz, as the underlying api assumes MHz

        frequencies = np.array(frequencies) * api.Hz

        # Update channel frequencies
        self.select_dds(channel)
        self.start_programming(api.FREQ_REGS)
        for frequency in frequencies:
            error_parse(api.pb_set_freq)(frequency)
        self.stop_programming()

    def set_phases(self, channel, phases):
        """ Sets the DDS phase for the specified channel and register

        Args:
            phases (list(double)): List of phases for a channel register
            channel        (int) : Either DDS0 (0) or DDS1 (1)
        """
        self.select_dds(channel)
        self.start_programming(self.TX_PHASE_REGS)
        for phase in phases:
            error_parse(api.pb_set_phase)(phase)
        self.stop_programming()

    def set_amplitudes(self, channel, amplitudes):
        """ Sets the DDS amplitude for the specified channel and register

        Args:
            channel (int) : Either DDS0 (0) or DDS1 (1)
            amplitudes (list(double)): Register amplitudes in Volt for a 
                channel 
        """
        self.select_dds(channel)
        # Amplitudes does not need to start programming
        for register, amplitude in enumerate(amplitudes):
            # Looking at the DDS output, an amplitude of 1 apparently
            # corresponds to 0.6V output. This further decreases if the
            # frequency is below 1 MHz,  but above 1MHz it is fairly constant.
            amplitude /= 0.6
            error_parse(api.pb_set_amp)(amplitude, register)

