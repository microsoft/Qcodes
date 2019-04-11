from qcodes import Instrument
from qcodes.utils.validators import Ints, Numbers
import ctypes
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ShamrockCIF:
    """
    Wrapper class for the ShamrockCIF.dll library.

    The class has been tested for a Shamrock SR750.

    Args:
        dll_path: Path to the ShamrockCIF.dll file. If not set, a default path is used.
        verbose: Flag for the verbose behaviour. If true, successful events are printed.

    Attributes:
        verbose: Flag for the verbose behaviour.
        dll: WinDLL object for ShamrockCIF.dll.

    """

    # default dll path
    _dll_path = 'C:\\Program Files\\Andor SDK\\Shamrock64\\ShamrockCIF.dll'

    # success and error codes
    _success_codes = {20202: 'SHAMROCK_SUCCESS'}
    _error_codes = {
        20201: 'SHAMROCK_COMMUNICATION_ERROR',
        20266: 'SHAMROCK_P1INVALID',
        20267: 'SHAMROCK_P2INVALID',
        20268: 'SHAMROCK_P3INVALID',
        20269: 'SHAMROCK_P4INVALID',
        20270: 'SHAMROCK_P5INVALID',
        20275: 'SHAMROCK_NOT_INITIALIZED',
        20292: 'SHAMROCK_NOT_AVAILABLE'
    }

    def __init__(self, dll_path: Optional[str] = None, verbose: Optional[bool] = False):

        # save attributes
        self.verbose: bool = verbose

        # connect to the DLL
        current_path = os.getcwd()
        os.chdir(os.path.dirname(self._dll_path))
        self.dll: ctypes.WinDLL = ctypes.windll.LoadLibrary(dll_path or self._dll_path)
        os.chdir(current_path)

    def error_check(self, code, function_name=''):
        if code in self._success_codes.keys():
            if self.verbose:
                logger.info(f"ShamrockCIF: [{function_name}]: {self._success_codes[code]}")
        elif code in self._error_codes.keys():
            error_string = f"ShamrockCIF: [{function_name}]: {self._error_codes[code]}"
            logger.error(error_string)
            raise Exception(error_string)
        else:
            error_string = f"ShamrockCIF: [{function_name}]: Unknown code: {code}"
            logger.error(error_string)
            raise Exception(error_string)

    # dll functions
    def close(self):
        code = self.dll.ShamrockClose()
        self.error_check(code, 'ShamrockClose')

    def get_calibration(self, device, number_pixels):
        c_device = ctypes.c_int(device)
        c_calibration = (ctypes.c_float * number_pixels)()
        c_number_pixels = ctypes.c_int(number_pixels)
        code = self.dll.ShamrockGetCalibration(c_device, c_calibration, c_number_pixels)
        self.error_check(code, 'ShamrockGetCalibration')
        return [max(0.0, v) for v in c_calibration]

    def get_grating(self, device):
        c_device = ctypes.c_int(device)
        c_grating = ctypes.c_int()
        code = self.dll.ShamrockGetGrating(c_device, ctypes.byref(c_grating))
        self.error_check(code, 'ShamrockGetGrating')
        return c_grating.value

    def get_grating_info(self, device, grating):
        c_device = ctypes.c_int(device)
        c_grating = ctypes.c_int(grating)
        c_lines = ctypes.c_float()
        c_blaze = ctypes.create_string_buffer(64)
        c_home = ctypes.c_int()
        c_offset = ctypes.c_int()
        code = self.dll.ShamrockGetGratingInfo(c_device, c_grating, ctypes.byref(c_lines), c_blaze, ctypes.byref(c_home), ctypes.byref(c_offset))
        self.error_check(code, 'ShamrockGetGratingInfo')
        return c_lines.value, c_blaze.value, c_home.value, c_offset.value

    def get_number_devices(self):
        c_no_devices = ctypes.c_int()
        code = self.dll.ShamrockGetNumberDevices(ctypes.byref(c_no_devices))
        self.error_check(code, 'ShamrockGetNumberDevices')
        return c_no_devices.value

    def get_number_gratings(self, device):
        c_device = ctypes.c_int(device)
        c_no_gratings = ctypes.c_int()
        code = self.dll.ShamrockGetNumberGratings(c_device, ctypes.byref(c_no_gratings))
        self.error_check(code, 'ShamrockGetNumberGratings')
        return c_no_gratings.value

    def get_number_pixels(self, device):
        c_device = ctypes.c_int(device)
        c_number_pixels = ctypes.c_int()
        code = self.dll.ShamrockGetNumberPixels(c_device, ctypes.byref(c_number_pixels))
        self.error_check(code, 'ShamrockGetNumberPixels')
        return c_number_pixels.value

    def get_pixel_width(self, device):
        c_device = ctypes.c_int(device)
        c_width = ctypes.c_float()
        code = self.dll.ShamrockGetPixelWidth(c_device, ctypes.byref(c_width))
        self.error_check(code, 'ShamrockGetPixelWidth')
        return c_width.value

    def get_serial_number(self, device):
        c_device = ctypes.c_int(device)
        c_serial = ctypes.create_string_buffer(128)
        code = self.dll.ShamrockGetSerialNumber(c_device, c_serial)
        self.error_check(code, 'ShamrockGetSerialNumber')
        return c_serial.value.decode('ascii')

    def get_slit(self, device):
        c_device = ctypes.c_int(device)
        c_width = ctypes.c_float()
        code = self.dll.ShamrockGetSlit(c_device, ctypes.byref(c_width))
        self.error_check(code, 'ShamrockGetSlit')
        return c_width.value

    def get_wavelength(self, device):
        c_device = ctypes.c_int(device)
        c_wavelength = ctypes.c_float()
        code = self.dll.ShamrockGetWavelength(c_device, ctypes.byref(c_wavelength))
        self.error_check(code, 'ShamrockGetWavelength')
        return c_wavelength.value

    def get_wavelength_limits(self, device, grating):
        c_device = ctypes.c_int(device)
        c_grating = ctypes.c_int(grating)
        c_min = ctypes.c_float()
        c_max = ctypes.c_float()
        code = self.dll.ShamrockGetWavelengthLimits(c_device, c_grating, ctypes.byref(c_min), ctypes.byref(c_max))
        self.error_check(code, 'ShamrockGetWavelengthLimits')
        return c_min.value, c_max.value

    def initialize(self):
        code = self.dll.ShamrockInitialize("")
        self.error_check(code, 'ShamrockInitialize')

    def set_grating(self, device, grating):
        c_device = ctypes.c_int(device)
        c_grating = ctypes.c_int(grating)
        code = self.dll.ShamrockSetGrating(c_device, c_grating)
        self.error_check(code, 'ShamrockSetGrating')

    def set_number_pixels(self, device, number_pixels):
        c_device = ctypes.c_int(device)
        c_number_pixels = ctypes.c_int(number_pixels)
        code = self.dll.ShamrockSetNumberPixels(c_device, c_number_pixels)
        self.error_check(code, 'ShamrockSetNumberPixels')

    def set_pixel_width(self, device, width):
        c_device = ctypes.c_int(device)
        c_width = ctypes.c_float(width)
        code = self.dll.ShamrockSetPixelWidth(c_device, c_width)
        self.error_check(code, 'ShamrockSetPixelWidth')

    def set_slit(self, device, width):
        c_device = ctypes.c_int(device)
        c_width = ctypes.c_float(width)
        code = self.dll.ShamrockSetSlit(c_device, c_width)
        self.error_check(code, 'ShamrockSetSlit')

    def set_wavelength(self, device, wavelength):
        c_device = ctypes.c_int(device)
        c_wavelength = ctypes.c_float(wavelength)
        code = self.dll.ShamrockSetWavelength(c_device, c_wavelength)
        self.error_check(code, 'ShamrockSetWavelength')


class Shamrock_SR750(Instrument):
    """
    Instrument driver for the Shamrock SR750 spectrometer.

    Args:
        name: Instrument name.
        dll_path: Path to the ShamrockCIF.dll file. If not set, a default path is used.
        device_id: ID for the desired spectrometer.
        ccd_number_pixels: Number of pixels on the connected CCD.
        ccd_pixel_width: Pixel width on the connected CCD.

    Attributes:
        ShamrockCIF: DLL wrapper for ShamrockCIF.dll
        device_id: Spectrometer device ID.
        serial_number: Serial number of the spectrometer.
        number_gratings: Number of gratings on the spectrometer.
    """

    def __init__(self, name: str, dll_path: Optional[str] = None, device_id: Optional[int] = 0, ccd_number_pixels: Optional[int] = 1024, ccd_pixel_width: Optional[int] = 26, **kwargs):

        super().__init__(name, **kwargs)

        # link to dll
        self.ShamrockCIF: ShamrockCIF = ShamrockCIF(dll_path=dll_path)

        # store device number
        self.device_id: int = device_id

        # initialize Shamrock
        self.ShamrockCIF.initialize()

        # read info from Shamrock
        self.serial_number: int = self.ShamrockCIF.get_serial_number(self.device_id)
        self.number_gratings: int = self.ShamrockCIF.get_number_gratings(self.device_id)

        # send CCD info to Shamrock
        self.ShamrockCIF.set_number_pixels(self.device_id, ccd_number_pixels)
        self.ShamrockCIF.set_pixel_width(self.device_id, ccd_pixel_width)

        # add the instrument parameters
        self.add_parameter('blaze',
                           get_cmd=self._get_blaze,
                           get_parser=int,
                           label='Blaze')

        self.add_parameter('calibration',
                           get_cmd=self._get_calibration,
                           unit='nm',
                           label='Calibration')

        self.add_parameter('grating',
                           get_cmd=self._get_grating,
                           set_cmd=self._set_grating,
                           get_parser=int,
                           vals=Ints(min_value=1,
                                     max_value=self.number_gratings),
                           label='Grating')

        self.add_parameter('groove_density',
                           get_cmd=self._get_groove_density,
                           get_parser=int,
                           unit='l/mm',
                           label='Groove density')

        self.add_parameter('slit',
                           get_cmd=self._get_slit,
                           set_cmd=self._set_slit,
                           get_parser=int,
                           vals=Ints(min_value=10,
                                     max_value=2500),
                           unit=u"\u03BC"+'m',
                           label='Slit')

        min_wavelength, max_wavelength = self.ShamrockCIF.get_wavelength_limits(self.device_id, self._get_grating())
        self.add_parameter('wavelength',
                           get_cmd=self._get_wavelength,
                           set_cmd=self._set_wavelength,
                           get_parser=float,
                           vals=Numbers(min_value=min_wavelength,
                                        max_value=max_wavelength),
                           unit='nm',
                           label='Wavelength')

        # print connect message
        self.connect_message()

    # get methods

    def _get_blaze(self):
        grating = self.ShamrockCIF.get_grating(self.device_id)
        grating_info = self.ShamrockCIF.get_grating_info(self.device_id, grating)
        return grating_info[1]

    def _get_calibration(self):
        return self.ShamrockCIF.get_calibration(self.device_id, 1024)

    def _get_grating(self):
        return self.ShamrockCIF.get_grating(self.device_id)

    def _get_groove_density(self):
        grating = self.ShamrockCIF.get_grating(self.device_id)
        grating_info = self.ShamrockCIF.get_grating_info(self.device_id, grating)
        return grating_info[0]

    def get_idn(self):
        return {'vendor': 'Shamrock', 'serial': self.serial_number}

    def _get_slit(self):
        return self.ShamrockCIF.get_slit(self.device_id)

    def _get_wavelength(self):
        return self.ShamrockCIF.get_wavelength(self.device_id)

    # set methods

    def _set_grating(self, grating):
        self.ShamrockCIF.set_grating(self.device_id, grating)
        min_wavelength, max_wavelength = self.ShamrockCIF.get_wavelength_limits(self.device_id, grating)
        self.wavelength.vals = Numbers(min_value=min_wavelength, max_value=max_wavelength)

    def _set_slit(self, val):
        self.ShamrockCIF.set_slit(self.device_id, val)

    def _set_wavelength(self, wavelength):
        self.ShamrockCIF.set_wavelength(self.device_id, wavelength)

    # further methods

    def close(self):
        self.ShamrockCIF.close()
        super().close()
