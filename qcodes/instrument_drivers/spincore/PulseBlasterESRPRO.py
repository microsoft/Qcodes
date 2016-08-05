import qcodes as qc
import ctypes

from qcodes import Instrument
from qcodes.utils.validators import Numbers, Ints, Enum, Strings
from qcodes.instrument.parameter import ManualParameter

from qcodes.instrument_drivers.spincore import spinapi as api

class PulseBlaster(Instrument):

    instructions = {'CONTINUE', 0,  #inst_data=Not used
                    'STOP', 1,      #inst_data=Not used
                    'LOOP', 2,      #inst_data=Number of desired loops
                    'END_LOOP', 3,  #inst_data=Address of instruction originating loop
                    'JSR', 4,       #inst_data=Address of first instruction in subroutine
                    'RTS', 5,       #inst_data=Not Used
                    'BRANCH', 6,    #inst_data=Address of instruction to branch to
                    'LONG_DELAY', 7,#inst_data=Number of desired repetitions
                    'WAIT', 8}      #inst_data=Not used

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

        self.add_parameter('core_clock',
                           label='Core clock',
                           units='MHz',
                           set_cmd=api.pb_core_clock,
                           vals=Numbers(0, 500))  # Not sure what the range is

        self.add_function('select_board',
                          call_cmd=api.pb_select_board, args=[Enum(0, 1, 2, 3, 4)])

        self.add_function('initialize',
                          call_cmd=self.initialize)

        self.add_function('start',
                          call_cmd=api.pb_start)

        self.add_function('start_programming',
                          call_cmd=self.start_programming, args=[Numbers(0,4)])

        self.add_function('send_instruction',
                          call_cmd=self.send_instruction,
                          args=[Ints(), Strings(), Ints(), Ints()])

        self.add_function('stop',
                          call_cmd=api.pb_stop)

        self.add_function('close',
                          call_cmd=self.close)

    def initialize(self):
        if api.pb_init() != 0:
            print(api.pb_init())
            raise IOError("Error initializing board: %s" % api.pb_get_error())

    def start_programming(self, device):
        if api.pb_start_programming(device) != 0:
            raise IOError("Error starting programming: %s" % api.pb_get_error())

    def send_instruction(self, flags, instruction, inst_data, length):
        instruction_int = self.instructions[instruction]
        #Need to call underlying spinapi because function does not exist in wrapper
        api.spinapi.pb_inst_pbonly(ctypes.c_uint64(flags),
                                   ctypes.c_int(instruction_int),
                                   ctypes.c_int(inst_data),
                                   ctypes.c_double(length))

    def close(self):
        #Terminate communication from api
        api.pb_close()
        super().close()