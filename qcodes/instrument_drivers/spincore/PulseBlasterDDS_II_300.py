from functools import partial, wraps
import logging

from qcodes.instrument.base import Instrument
from qcodes.utils.validators import Numbers, Enum, Ints, Strings, Anything

try:
    from . import spinapi as api
except:
    raise ImportError('To use a SpinCore PulseBlaster, install the Python SpinAPI')


logger = logging.getLogger(__name__)

# Add seconds to list of Pulseblaster keywords
api.s = 1000000000.0
api.TX_PHASE_REGS  = 2
api.RX_PHASE_REGS  = 3


def error_parse(f):
    @wraps(f)
    def error_wrapper(*args, **kwargs):
        value = f(*args, **kwargs)
        if not isinstance(value, str) and value < 0:
            raise IOError('{}: {}'.format(value, api.pb_get_error()))
        return value
    return error_wrapper


class PulseBlaster_DDS(Instrument):
    """
    This is the qcodes driver for the SpinCore PulseBlasterDDS-II-300
    """

    # pb_start_programming options
    TX_PHASE_REGS  = 2
    RX_PHASE_REGS  = 3
    #COS_PHASE_REGS = 4 # RadioProcessor boards ONLY
    #SIN_PHASE_REGS = 5 # RadioProcessor boards ONLY

    # pb_write_register options
    BASE_ADDR       = 0x40000
    FLAG_STATES     = BASE_ADDR + 0x8
    START_LOCATION  = BASE_ADDR + 0x7

    # pb_dds_load options
    DEVICE_DDS   = 0x099000
    DEVICE_SHAPE = 0x099001

    N_CHANNELS = 2
    N_FREQ_REGS = 16
    N_PHASE_REGS = 8
    N_AMPLITUDE_REGS = 4

    DEFAULT_DDS_INST = (0, 0, 0, 0, 0)

    def __init__(self, name,  **kwargs):
        """
        Initialize the pulseblaster DDS
        
        Args:
            name: Name of instrument 
            **kwargs: Additional keyword arguments passed to Instrument
        """
        super().__init__(name, **kwargs)

        # Initialise the DDS
        # NOTE: Hard coded value for board may want to be part of initialize
        if self.count_boards() > 0:
            self.initialize()
            self.select_board(0)
            # Call set defaults to give board a well defined state
            api.pb_set_defaults()
        else:
            raise IOError("PB Error: Can't find board")

        # Internal State Variables, used when setting individual registers when
        # others may be undefined
        self._frequency = [[0.0] * self.N_FREQ_REGS,  [0.0] * self.N_FREQ_REGS]
        self._phase     = [[0.0] * self.N_PHASE_REGS, [0.0] * self.N_PHASE_REGS]

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

        for n in range(self.N_CHANNELS):
            # DDS Register Bank
            for r in range(self.N_FREQ_REGS):
                self.add_parameter(
                    'frequency_n{}_r{}'.format(n, r),
                    label='DDS Frequency for channel {}, '
                          'register {}'.format(n, r),
                    set_cmd=partial(self.set_frequency_register,
                                    channel=n, register=r),
                    vals=Numbers(),
                    docstring='')

            for r in range(self.N_PHASE_REGS):
                self.add_parameter(
                    'phase_n{}_r{}'.format(n, r),
                    label='DDS Phase for channel {}, '
                          'register {}'.format(n, r),
                    set_cmd=partial(self.set_phase_register,
                                    channel=n, register=r),
                    vals=Numbers(),
                    docstring='')

            for r in range(self.N_AMPLITUDE_REGS):
                self.add_parameter(
                    'amplitude_n{}_r{}'.format(n, r),
                    label='DDS Amplitude for channel {}, '
                          'register {}'.format(n, r),
                    set_cmd=partial(self.set_amplitude_register,
                                    channel=n, register=r),
                    vals=Numbers(),
                    docstring='')

    ###########################################################################
    ###                         DDS Board commands                          ###
    ###########################################################################
    # Just wrapped from spinapi.py #

    @staticmethod
    def get_error():
        """ Print library error as UTF-8 encoded string. """
        return str(api.pb_get_error())

    @error_parse
    def get_version(self):
        """ Return the current version of the spinapi library being used. """
        return api.pb_get_version()

    @error_parse
    def count_boards(self):
        """ Print the number of boards detected in the system. """
        return api.pb_count_boards()

    @error_parse
    def initialize(self):
        """ Initialize currently selected board. """
        return api.pb_init()

    @error_parse
    def set_debug(self, debug):
        """ Enables logging to a log.txt file in the current directory """
        return api.pb_set_debug(debug)

    @error_parse
    def set_defaults(self):
        """ Set board defaults. Must be called before using any other board 
        functions."""
        return api.pb_set_defaults()

    def set_core_clock(self, clock):
        """ Sets the core clock reference value
        
        Note: This does not change anything in the device, it just provides 
        the library with the value given
        """
        api.pb_core_clock(clock)

    @error_parse
    def write_register(self, address, value):
        """ Write to one of two core registers in the DDS
        
        Args:
            address (int) : the address of the register
            value   (int) : the value to be written to the register
        """
        return api.pb_write_register(address, value)

    ###########################################################################
    ###                        DDS Control commands                         ###
    ###########################################################################

    @error_parse
    def start(self):
        return api.pb_start()

    @error_parse
    def reset(self):
        return api.pb_reset()

    @error_parse
    def stop(self):
        return api.pb_stop()

    @error_parse
    def close(self):
        return api.pb_close()

    def add_instruction(self, inst, channel):
        self.instructions[channel].append(inst)

    @error_parse
    def start_programming(self, mode):
        """ Start a programming sequence 
        
        Args:
            mode    (int) : one of (PULSE_PROGRAM, FREQ_REGS, etc.)
        """
        return api.pb_start_programming(mode)

    @error_parse
    def inst_dds2(self, inst):
        """ During start_programming(PULSE_PROGRAM) 
            add an unshaped sine pulse to program
        
        Args:
            inst  (tuple) : a tuple to program the board in the
                            (FREQ0, PHASE0, ...)
        """
        inst = inst[:-1] + (inst[-1] * api.s,)
        return api.pb_inst_dds2(*inst)

    @error_parse
    def inst_dds2_shape(self, inst):
        """ During start_programming(PULSE_PROGRAM) 
            add a shaped pulse to program, if desired
        
        Args:
            inst  (tuple) : a tuple to program the board in the
                            (FREQ0, PHASE0, ...)
        """
        inst = inst[:-1] + (inst[-1] * api.s,)
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
        return api.pb_dds_load(device)

    @error_parse
    def stop_programming(self):
        """ End a programming sequence """
        return api.pb_stop_programming()

    @error_parse
    def select_dds(self, channel):
        return api.pb_select_dds(channel)

    def program_pulse_sequence(self, pulse_sequence):
        """ An all-in-one function to send a pulse sequence to the board

        Args:
            pulse_sequence (list) : a list of instructions to program the board
                                    the instructions should be tuples i.e.
                                    (FREQ0, PHASE0, ...)
        """
        self.start_programming(api.PULSE_PROGRAM)
        for p in pulse_sequence:
            # convert last element from seconds to ns
            p = p[:-1] + (p[-1]*api.s,)
            # * breaks tuple into args, 
            self.inst_dds2(*p)
        self.stop_programming()

    def program_inst(self):
        """ Send the current instruction list to the board
        
        Args:
            
        """
        raise NotImplementedError('Warning: Unimplemented function')
        # pb_start_programming()
        # n_inst = 0
        # # Find the longest set of instructions defined
        # for n in range(self.N_CHANNELS):
        #    n_inst = max(n_inst, len(self.__chan_instructions[n]))
        # for i in range(n_inst):
        #    # TODO: use lambdas here for ease
        #    #if (self.__chan_instructions[
        #    pass
        # pb_stop_programming()

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

    def set_frequency_register(self, frequency, channel, register):
        """ Sets the DDS frequency for the specified channel and register
        
        Args:
            frequency (double) : the frequency in Hz to write to the register
            register     (int) : the register number
            channel      (int) : Either DDS0 (0) or DDS1 (1)
        """
        # Scale the frequency to Hertz, as the underlying api assumes MHz
        frequency *= api.Hz
        self._frequency[channel][register] = frequency

        # Update channel frequencies
        self.select_dds(channel)
        self.start_programming(api.FREQ_REGS)
        for frequency in self._frequency[channel]:
            error_parse(api.pb_set_freq(frequency))
        self.stop_programming()

    def set_phase_register(self, phase, channel, register):
        """ Sets the DDS phase for the specified channel and register
        
        Args:
            phase     (double) : the phase in degrees to write to the register
            register     (int) : the register number
            channel      (int) : Either DDS0 (0) or DDS1 (1)
        """
        self._phase[channel][register] = phase

        self.select_dds(channel)
        self.start_programming(self.TX_PHASE_REGS)
        for phase in self._phase[channel]:
            error_parse(api.pb_set_phase(phase))
        self.stop_programming()

    @error_parse
    def set_amplitude_register(self, amplitude, channel, register):
        """ Sets the DDS amplitude for the specified channel and register
        
        Args:
            amplitude (double) : the amplitude in volts to write to the register
            register     (int) : the register number
            channel      (int) : Either DDS0 (0) or DDS1 (1)
        """
        self.select_dds(channel)
        return api.pb_set_amp(amplitude, register)

