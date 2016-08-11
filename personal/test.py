import qcodes as qc
from qcodes import Instrument
from qcodes.utils.validators import Numbers, Ints, Enum, Strings
from time import sleep

if __name__ == "__main__":
    import clr

    # from qcodes.instrument_drivers.spincore import PulseBlasterESRPRO
    #
    # PulseBlaster = PulseBlasterESRPRO.PulseBlaster('PulseBlaster', server_name=None)
    # PulseBlaster.detect_boards()
    # PulseBlaster.select_board(0)
    # PulseBlaster.core_clock(500)
    #
    # N = 1000
    # PulseBlaster.start_programming()
    # start = PulseBlaster.send_instruction(15,'continue',0,1000)
    # for k in range(16):
    #     PulseBlaster.send_instruction(k,'continue',0,1000)
    # PulseBlaster.send_instruction(0, 'branch', start, 1000)
    # PulseBlaster.stop_programming()
    #
    # sleep(1)
    # PulseBlaster.start()
    # sleep(5)
    # PulseBlaster.stop()