import ctypes


class Thorlabs_APT:
    """
    Wrapper class for the APT.dll Thorlabs APT Server library.
    The class has been tested for a Thorlabs MFF10x mirror flipper and a Thorlabs PRM1Z8 Polarizer Wheel.
    Args:
        dll_path (str): Path to the APT.dll file. If not set, a default path is used.
        verbose (bool): Flag for the verbose behaviour. If true, successful events are printed.
        event_dialog (bool): Flag for the event dialog. If true, event dialog pops up for information.
    Attributes:
        verbose (bool): Flag for the verbose behaviour.
        dll (WinDLL): WinDLL object for APT.dll.
    """

    # default dll path
    _dll_path = 'C:\\Program Files\\Thorlabs\\APT\\APT Server\\APT.dll'

    # success and error codes
    _success_code = 0
    _error_codes = {}

    def __init__(self, dll_path=None, verbose=False, event_dialog=False):

        # save attributes
        self.verbose = verbose

        # connect to the DLL
        self.dll = ctypes.CDLL(dll_path or self._dll_path)

        # initialize APT server
        self.apt_init()
        self.enable_event_dlg(event_dialog)

    def error_check(self, code, function_name=''):
        if code == self._success_code:
            if self.verbose:
                print("APT: [%s]: %s" % (function_name, 'OK - no error'))
        elif code in self._error_codes.keys():
            print("APT: [%s]: %s" % (function_name, self._error_codes[code]))
            raise Exception(self._error_codes[code])
        else:
            print("APT: [%s]: Unknown code: %s" % (function_name, code))
            raise Exception()

    def apt_clean_up(self):
        code = self.dll.APTCleanUp()
        self.error_check(code, 'APTCleanUp')

    def apt_init(self):
        code = self.dll.APTInit()
        self.error_check(code, 'APTInit')

    def enable_event_dlg(self, enable):
        c_enable = ctypes.c_bool(enable)
        code = self.dll.EnableEventDlg(c_enable)
        self.error_check(code, 'EnableEventDlg')

    def get_hw_info(self, serial_number):
        c_serial_number = ctypes.c_long(serial_number)
        c_sz_model = ctypes.create_string_buffer(64)
        c_sz_sw_ver = ctypes.create_string_buffer(64)
        c_sz_hw_notes = ctypes.create_string_buffer(64)
        code = self.dll.GetHWInfo(c_serial_number, c_sz_model, 64, c_sz_sw_ver, 64, c_sz_hw_notes, 64)
        self.error_check(code, 'GetHWInfo')
        return c_sz_model.value, c_sz_sw_ver.value, c_sz_hw_notes.value

    def get_hw_serial_num_ex(self, hw_type, index):
        c_hw_type = ctypes.c_long(hw_type)
        c_index = ctypes.c_long(index)
        c_serial_number = ctypes.c_long()
        code = self.dll.GetHWSerialNumEx(c_hw_type, c_index, ctypes.byref(c_serial_number))
        self.error_check(code, 'GetHWSerialNumEx')
        return c_serial_number.value

    def init_hw_device(self, serial_number):
        c_serial_number = ctypes.c_long(serial_number)
        code = self.dll.InitHWDevice(c_serial_number)
        self.error_check(code, 'InitHWDevice')

    def mot_get_position(self, serial_number):
        c_serial_number = ctypes.c_long(serial_number)
        c_position = ctypes.c_float()
        code = self.dll.MOT_GetPosition(c_serial_number, ctypes.byref(c_position))
        self.error_check(code, 'MOT_GetPosition')
        return c_position.value

    def mot_get_status_bits(self, serial_number):
        c_serial_number = ctypes.c_long(serial_number)
        c_status_bits = ctypes.c_long()
        code = self.dll.MOT_GetStatusBits(c_serial_number, ctypes.byref(c_status_bits))
        self.error_check(code, 'MOT_GetStatusBits')
        return c_status_bits.value

    def mot_move_absolute_ex(self, serial_number, absolute_position, wait):
        c_serial_number = ctypes.c_long(serial_number)
        c_absolute_position = ctypes.c_float(absolute_position)
        c_wait = ctypes.c_bool(wait)
        code = self.dll.MOT_MoveAbsoluteEx(c_serial_number, c_absolute_position, c_wait)
        self.error_check(code, 'MOT_MoveAbsoluteEx')

    def mot_move_jog(self, serial_number, direction, wait):
        c_serial_number = ctypes.c_long(serial_number)
        c_direction = ctypes.c_long(direction)
        c_wait = ctypes.c_bool(wait)
        code = self.dll.MOT_MoveJog(c_serial_number, c_direction, c_wait)
        self.error_check(code, 'MOT_MoveJog')
