from qcodes import Instrument, Parameter
from qcodes.utils.validators import Ints
from qcodes.utils.helpers import create_on_off_val_mapping
import ctypes
import time


class atmcd64d:
    """
    Wrapper class for the atmcd64.dll Andor library.

    The class has been tested for an Andor iDus DU401 BU2.

    Args:
        dll_path (str): Path to the atmcd64.dll file. If not set, a default path is used.
        verbose (bool): Flag for the verbose behaviour. If true, successful events are printed.

    Attributes:
        verbose (bool): Flag for the verbose behaviour.
        dll (WinDLL): WinDLL object for atmcd64.dll.

    """

    # default dll path
    _dll_path = 'C:\\Program Files\\Andor SDK\\atmcd64d.dll'

    # success and error codes
    _success_codes = {20002: 'DRV_SUCCESS', 20035: 'DRV_TEMP_NOT_STABILIZED', 20036: 'DRV_TEMPERATURE_STABILIZED',
                      20037: 'DRV_TEMPERATURE_NOT_REACHED'}
    _error_codes = {
        20001: 'DRV_ERROR_CODES', 20003: 'DRV_VXDNOTINSTALLED', 20004: 'DRV_ERROR_SCAN', 20005: 'DRV_ERROR_CHECK_SUM',
        20006: 'DRV_ERROR_FILELOAD', 20007: 'DRV_UNKNOWN_FUNCTION', 20008: 'DRV_ERROR_VXD_INIT',
        20009: 'DRV_ERROR_ADDRESS', 20010: 'DRV_ERROR_PAGELOCK', 20011: 'DRV_ERROR_PAGE_UNLOCK',
        20012: 'DRV_ERROR_BOARDTEST', 20013: 'DRV_ERROR_ACK', 20014: 'DRV_ERROR_UP_FIFO', 20015: 'DRV_ERROR_PATTERN',
        20017: 'DRV_ACQUISITION_ERRORS', 20018: 'DRV_ACQ_BUFFER', 20019: 'DRV_ACQ_DOWNFIFO_FULL',
        20020: 'DRV_PROC_UNKNOWN_INSTRUCTION', 20021: 'DRV_ILLEGAL_OP_CODE', 20022: 'DRV_KINETIC_TIME_NOT_MET',
        20023: 'DRV_ACCUM_TIME_NOT_MET', 20024: 'DRV_NO_NEW_DATA', 20026: 'DRV_SPOOLERROR',
        20027: 'DRV_SPOOLSETUPERROR', 20033: 'DRV_TEMPERATURE_CODES', 20034: 'DRV_TEMPERATURE_OFF',
        20038: 'DRV_TEMPERATURE_OUT_RANGE', 20039: 'DRV_TEMPERATURE_NOT_SUPPORTED', 20040: 'DRV_TEMPERATURE_DRIFT',
        20049: 'DRV_GENERAL_ERRORS', 20050: 'DRV_INVALID_AUX', 20051: 'DRV_COF_NOTLOADED', 20052: 'DRV_FPGAPROG',
        20053: 'DRV_FLEXERROR', 20054: 'DRV_GPIBERROR', 20064: 'DRV_DATATYPE', 20065: 'DRV_DRIVER_ERRORS',
        20066: 'DRV_P1INVALID', 20067: 'DRV_P2INVALID', 20068: 'DRV_P3INVALID', 20069: 'DRV_P4INVALID',
        20070: 'DRV_INIERROR', 20071: 'DRV_COFERROR', 20072: 'DRV_ACQUIRING', 20073: 'DRV_IDLE',
        20074: 'DRV_TEMPCYCLE', 20075: 'DRV_NOT_INITIALIZED', 20076: 'DRV_P5INVALID', 20077: 'DRV_P6INVALID',
        20078: 'DRV_INVALID_MODE', 20079: 'DRV_INVALID_FILTER', 20080: 'DRV_I2CERRORS',
        20081: 'DRV_DRV_I2CDEVNOTFOUND', 20082: 'DRV_I2CTIMEOUT', 20083: 'DRV_P7INVALID', 20089: 'DRV_USBERROR',
        20090: 'DRV_IOCERROR', 20091: 'DRV_VRMVERSIONERROR', 20093: 'DRV_USB_INTERRUPT_ENDPOINT_ERROR',
        20094: 'DRV_RANDOM_TRACK_ERROR', 20095: 'DRV_INVALID_TRIGGER_MODE', 20096: 'DRV_LOAD_FIRMWARE_ERROR',
        20097: 'DRV_DIVIDE_BY_ZERO_ERROR', 20098: 'DRV_INVALID_RINGEXPOSURES', 20990: 'DRV_ERROR_NOCAMERA',
        20991: 'DRV_NOT_SUPPORTED', 20992: 'DRV_NOT_AVAILABLE', 20115: 'DRV_ERROR_MAP', 20116: 'DRV_ERROR_UNMAP',
        20117: 'DRV_ERROR_MDL', 20118: 'DRV_ERROR_UNMDL', 20119: 'DRV_ERROR_BUFFSIZE', 20121: 'DRV_ERROR_NOHANDLE',
        20130: 'DRV_GATING_NOT_AVAILABLE', 20131: 'DRV_FPGA_VOLTAGE_ERROR', 20099: 'DRV_BINNING_ERROR',
        20100: 'DRV_INVALID_AMPLIFIER', 20101: 'DRV_INVALID_COUNTCONVERT_MODE'}

    def __init__(self, dll_path=None, verbose=False):

        self.verbose = verbose
        self.dll = ctypes.windll.LoadLibrary(dll_path or self._dll_path)

    def error_check(self, code, function_name=''):
        if code in self._success_codes.keys():
            if self.verbose:
                print("atmcd64d: [%s]: %s" % (function_name, self._success_codes[code]))
        elif code in self._error_codes.keys():
            print("atmcd64d: [%s]: %s" % (function_name, self._error_codes[code]))
            raise Exception(self._error_codes[code])
        else:
            print("atmcd64d: [%s]: Unknown code: %s" % (function_name, code))
            raise Exception()

    def cooler_off(self):
        code = self.dll.CoolerOFF()
        self.error_check(code, 'CoolerOFF')

    def cooler_on(self):
        code = self.dll.CoolerON()
        self.error_check(code, 'CoolerON')

    def get_acquired_data(self, size):
        c_data_array = ctypes.c_int * size
        c_data = c_data_array()
        code = self.dll.GetAcquiredData(ctypes.pointer(c_data), size)
        self.error_check(code, 'GetAcquiredData')
        acquired_data = []
        for i in range(len(c_data)):
            acquired_data.append(c_data[i])
        return acquired_data

    def get_acquisition_timings(self):
        c_exposure = ctypes.c_float()
        c_accumulate = ctypes.c_float()
        c_kinetic = ctypes.c_float()
        code = self.dll.GetAcquisitionTimings(ctypes.byref(c_exposure), ctypes.byref(c_accumulate), ctypes.byref(c_kinetic))
        self.error_check(code, 'GetAcquisitionTimings')
        return c_exposure.value, c_accumulate.value, c_kinetic.value

    def get_camera_handle(self, camera_index):
        c_camera_handle = ctypes.c_long()
        code = self.dll.GetCameraHandle(camera_index, ctypes.byref(c_camera_handle))
        self.error_check(code, 'GetCameraHandle')
        return c_camera_handle.value

    def get_camera_serial_number(self):
        c_serial_number = ctypes.c_int()
        code = self.dll.GetCameraSerialNumber(ctypes.byref(c_serial_number))
        self.error_check(code, 'GetCameraSerialNumber')
        return c_serial_number.value

    def get_hardware_version(self):
        c_pcb = ctypes.c_int()
        c_decode = ctypes.c_int()
        c_dummy1 = ctypes.c_int()
        c_dummy2 = ctypes.c_int()
        c_firmware_version = ctypes.c_int()
        c_firmware_build = ctypes.c_int()
        code = self.dll.GetHardwareVersion(ctypes.byref(c_pcb), ctypes.byref(c_decode), ctypes.byref(c_dummy1),
                                           ctypes.byref(c_dummy2), ctypes.byref(c_firmware_version),
                                           ctypes.byref(c_firmware_build))
        self.error_check(code)
        return c_pcb.value, c_decode.value, c_dummy1.value, c_dummy2.value, c_firmware_version.value, \
            c_firmware_build.value

    def get_head_model(self):
        c_head_model = ctypes.create_string_buffer(128)
        code = self.dll.GetHeadModel(c_head_model)
        self.error_check(code)
        return c_head_model.value.decode('ascii')

    def get_detector(self):
        c_x_pixels = ctypes.c_int()
        c_y_pixels = ctypes.c_int()
        code = self.dll.GetDetector(ctypes.byref(c_x_pixels), ctypes.byref(c_y_pixels))
        self.error_check(code, 'GetDetector')
        return c_x_pixels.value, c_y_pixels.value

    def get_filter_mode(self):
        c_mode = ctypes.c_int()
        code = self.dll.GetFilterMode(ctypes.byref(c_mode))
        self.error_check(code, 'GetFilterMode')
        return c_mode.value

    def get_status(self):
        c_status = ctypes.c_int()
        code = self.dll.GetStatus(ctypes.byref(c_status))
        self.error_check(code, 'GetStatus')
        return c_status.value

    def get_temperature(self):
        c_temperature = ctypes.c_int()
        code = self.dll.GetTemperature(ctypes.byref(c_temperature))
        self.error_check(code, 'GetTemperature')
        return c_temperature.value

    def get_temperature_range(self):
        c_min_temp = ctypes.c_int()
        c_max_temp = ctypes.c_int()
        code = self.dll.GetTemperatureRange(ctypes.byref(c_min_temp), ctypes.byref(c_max_temp))
        self.error_check(code, 'GetTemperatureRange')
        return c_min_temp.value, c_max_temp.value

    def initialize(self, directory):
        code = self.dll.Initialize(directory)
        self.error_check(code, 'Initialize')

    def is_cooler_on(self):
        c_cooler_status = ctypes.c_int()
        code = self.dll.IsCoolerOn(ctypes.byref(c_cooler_status))
        self.error_check(code, 'IsCoolerOn')
        return c_cooler_status.value

    def set_accumulation_cycle_time(self, cycle_time):
        c_cycle_time = ctypes.c_float(cycle_time)
        code = self.dll.SetAccumulationCycleTime(c_cycle_time)
        self.error_check(code, 'SetAccumulationCycleTime')

    def set_acqiusition_mode(self, mode):
        c_mode = ctypes.c_int(mode)
        code = self.dll.SetAcquisitionMode(c_mode)
        self.error_check(code, 'SetAcquisitionMode')

    def set_current_camera(self, camera_handle):
        c_camera_handle = ctypes.c_long(camera_handle)
        code = self.dll.SetCurrentCamera(c_camera_handle)
        self.error_check(code, 'SetCurrentCamera')

    def set_exposure_time(self, exposure_time):
        c_time = ctypes.c_float(exposure_time)
        code = self.dll.SetExposureTime(c_time)
        self.error_check(code, 'SetExposureTime')

    def set_filter_mode(self, mode):
        c_mode = ctypes.c_int(mode)
        code = self.dll.SetFilterMode(c_mode)
        self.error_check(code, 'SetFilterMode')

    def set_number_accumulations(self, number):
        c_number = ctypes.c_int(number)
        code = self.dll.SetNumberAccumulations(c_number)
        self.error_check(code, 'SetNumberAccumulations')

    def set_read_mode(self, mode):
        code = self.dll.SetReadMode(mode)
        self.error_check(code, 'SetReadMode')

    def set_shutter(self, typ, mode, closing_time, opening_time):
        c_typ = ctypes.c_int(typ)
        c_mode = ctypes.c_int(mode)
        c_closing_time = ctypes.c_int(closing_time)
        c_opening_time = ctypes.c_int(opening_time)
        code = self.dll.SetShutter(c_typ, c_mode, c_closing_time, c_opening_time)
        self.error_check(code, 'SetShutter')

    def set_temperature(self, temperature):
        c_temperature = ctypes.c_int(temperature)
        code = self.dll.SetTemperature(c_temperature)
        self.error_check(code, 'SetTemperature')

    def set_trigger_mode(self, mode):
        c_mode = ctypes.c_int(mode)
        code = self.dll.SetTriggerMode(c_mode)
        self.error_check(code, 'SetTriggerMode')

    def shut_down(self):
        code = self.dll.ShutDown()
        self.error_check(code, 'ShutDown')

    def start_acquisition(self):
        code = self.dll.StartAcquisition()
        self.error_check(code, 'StartAcquisition')

    def wait_for_acquisition(self):
        code = self.dll.WaitForAcquisition()
        self.error_check(code, 'WaitForAcquisition')


class Spectrum(Parameter):
    """
    Parameter class for a spectrum taken with an Andor CCD.

    The spectrum is saved in a list with the length being set by the number of pixels on the CCD.

    Args:
        name (str): Parameter name.

    """

    def __init__(self, name, *args, **kwargs):

        self.ccd = kwargs['instrument']
        super().__init__(name, *args, **kwargs)

    def get_raw(self):

        # get acquisition mode
        acquisition_mode = self.ccd.acquisition_mode.get()

        # start acquisition
        self.ccd.atmcd64d.start_acquisition()

        # wait for single acquisition
        if acquisition_mode == 'single scan':
            self.ccd.atmcd64d.wait_for_acquisition()

        # wait for accumulate acquisition
        elif acquisition_mode == 'accumulate':
            number_accumulations = self.ccd.number_accumulations.get()
            for i in range(number_accumulations):
                self.ccd.atmcd64d.wait_for_acquisition()

        # get and return spectrum
        return self.ccd.atmcd64d.get_acquired_data(self.ccd.x_pixels)


class Andor_DU401(Instrument):
    """
    Instrument driver for the Andor DU401 BU2 CCD.

    Args:
        name (str): Instrument name.
        dll_path (str): Path to the atmcd64.dll file. If not set, a default path is used.
        camera_id (int): ID for the desired CCD.
        setup (bool): Flag for the setup of the CCD. If true, some default settings will be sent to the CCD.

    Attributes:
        serial_number (int): Serial number of the CCD.
        head_model (str): Head model of the CCD.
        firmware_version (int): Firmware version of the CCD.
        firmware_build (int): Firmware build of the CCD.
        x_pixels (int): Number of pixels on the x axis.
        y_pixels (int): Number of pixels on the y axis.

    """

    # TODO (SvenBo90): implement further acquisition modes
    # TODO (SvenBo90): implement further read modes
    # TODO (SvenBo90): implement further trigger modes
    # TODO (SvenBo90): add and delete parameters dynamically when switching acquisition mode or read mode
    # TODO (SvenBo90): handle shutter closing and opening timings

    def __init__(self, name, dll_path=None, camera_id=0, setup=True, **kwargs):

        super().__init__(name, **kwargs)

        # link to dll
        self.atmcd64d = atmcd64d(dll_path=dll_path)

        # initialization
        self.atmcd64d.initialize(' ')
        self.atmcd64d.set_current_camera(self.atmcd64d.get_camera_handle(camera_id))

        # get camera information
        self.serial_number = self.atmcd64d.get_camera_serial_number()
        self.head_model = self.atmcd64d.get_head_model()
        self.firmware_version = self.atmcd64d.get_hardware_version()[4]
        self.firmware_build = self.atmcd64d.get_hardware_version()[5]
        self.x_pixels, self.y_pixels = self.atmcd64d.get_detector()

        # add the instrument parameters
        def accumulation_cycle_time_parser(ans):
            return float(ans[1])
        self.add_parameter('accumulation_cycle_time',
                           get_cmd=self.atmcd64d.get_acquisition_timings,
                           set_cmd=self.atmcd64d.set_accumulation_cycle_time,
                           get_parser=accumulation_cycle_time_parser,
                           unit='s',
                           label='accumulation cycle time')

        self.add_parameter('acquisition_mode',
                           set_cmd=self.atmcd64d.set_acqiusition_mode,
                           val_mapping={
                               'single scan': 1,
                               'accumulate': 2
                           },
                           label='acquisition mode')

        self.add_parameter('cooler',
                           get_cmd=self.atmcd64d.is_cooler_on,
                           set_cmd=self.set_cooler,
                           val_mapping=create_on_off_val_mapping(on_val=1, off_val=0),
                           label='cooler')

        def exposure_time_parser(ans):
            return float(ans[0])
        self.add_parameter('exposure_time',
                           get_cmd=self.atmcd64d.get_acquisition_timings,
                           set_cmd=self.atmcd64d.set_exposure_time,
                           get_parser=exposure_time_parser,
                           unit='s',
                           label='exposure time')

        self.add_parameter('filter_mode',
                           get_cmd=self.atmcd64d.get_filter_mode,
                           set_cmd=self.atmcd64d.set_filter_mode,
                           val_mapping=create_on_off_val_mapping(on_val=2, off_val=0),
                           label='filter mode')

        self.add_parameter('number_accumulations',
                           set_cmd=self.atmcd64d.set_number_accumulations,
                           label='number accumulations')

        self.add_parameter('read_mode',
                           set_cmd=self.atmcd64d.set_read_mode,
                           val_mapping={'full vertical binning': 0})

        min_temperature, max_temperature = self.atmcd64d.get_temperature_range()
        self.add_parameter('set_temperature',
                           set_cmd=self.atmcd64d.set_temperature,
                           vals=Ints(min_value=min_temperature,
                                     max_value=max_temperature),
                           unit=u"\u00b0"+'C',
                           label='set temperature')

        self.add_parameter('shutter_mode',
                           set_cmd=self.set_shutter_mode,
                           val_mapping={
                               'fully auto': 0,
                               'permanently open': 1,
                               'permanently closed': 2},
                           label='shutter mode')

        self.add_parameter('spectrum',
                           parameter_class=Spectrum,
                           shape=(1, self.x_pixels),
                           label='spectrum')

        self.add_parameter('temperature',
                           get_cmd=self.atmcd64d.get_temperature,
                           unit=u"\u00b0"+'C',
                           label='temperature')

        self.add_parameter('trigger_mode',
                           set_cmd=self.atmcd64d.set_trigger_mode,
                           val_mapping={'internal': 0})

        # set up detector with default settings
        if setup:
            self.cooler.set(True)
            self.set_temperature.set(-60)
            self.read_mode.set('full vertical binning')
            self.acquisition_mode.set('single scan')
            self.trigger_mode.set('internal')
            self.shutter_mode.set('fully auto')

        # print connect message
        self.connect_message(idn_param='IDN')

    # get methods
    def get_idn(self):
        return {'vendor': 'Andor', 'model': self.head_model,
                'serial': self.serial_number, 'firmware': str(self.firmware_version)+'.'+str(self.firmware_build)}

    # set methods
    def set_cooler(self, cooler_on):
        if cooler_on == 1:
            self.atmcd64d.cooler_on()
        elif cooler_on == 0:
            self.atmcd64d.cooler_off()

    def set_shutter_mode(self, shutter_mode):
        self.atmcd64d.set_shutter(1, shutter_mode, 30, 30)

    # further methods
    def close(self):
        self.atmcd64d.shut_down()
        super().close()
