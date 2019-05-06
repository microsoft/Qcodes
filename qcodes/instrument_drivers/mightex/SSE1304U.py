""" This module provides a driver for the Mightex SSE-1304-U spectrometer. """

from pathlib import Path
import json
from typing import Union
import ctypes as ct

import numpy as np
import numpy.ctypeslib as npct

import qcodes as qc
import qcodes.utils.validators as vals

from .dllwrapper import MightexDLLWrapper, FrameRecord


class WavelengthAxis(qc.Parameter):
    """ Holds the wavelengths corresponding to CCD pixels. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not isinstance(self.root_instrument, MightexSSE_1304_U):
            raise ValueError('WavelengthAxis added to the wrong instrument '
                             'type. This parameter is only compatible with'
                             'MightexSSE_1304_U.')
        self.spectrometer: MightexSSE_1304_U = self.root_instrument

    def get_raw(self) -> np.ndarray:
        if self.spectrometer.calib is None:
            raise ValueError('No calibration loaded. Load a calibration '
                             'file prior to using WavelengthAxis.')

        return self.spectrometer.calib(np.arange(0, MightexSSE_1304_U.PIXELS))


class IntensitySpectrum(qc.ParameterWithSetpoints):
    """ Holds the measured intensity. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not isinstance(self.root_instrument, MightexSSE_1304_U):
            raise ValueError('WavelengthAxis added to the wrong instrument '
                             'type. This parameter is only compatible with'
                             'MightexSSE_1304_U.')
        self.spectrometer: MightexSSE_1304_U = self.root_instrument

    def get_raw(self) -> np.ndarray:
        """ Get an averaged frame with dark data subtracted. """
        avg = self.spectrometer.averaging()
        frames = []

        for i in range(avg):
            frame = self.spectrometer.get_single_frame()
            frames.append(frame - self.spectrometer.drkdata)

        return sum(frames) / avg


class MightexSSE_1304_U(qc.Instrument):
    """ Driver for the Mightex SSE-1304-U spectrometer. """

    PIXELS = 3648

    def __init__(self, name: str,
                 dll_path: Union[str, Path]=r'C:\Windows\System32',
                 dev_id: int=1, **kwargs):
        super().__init__(name, **kwargs)

        self._dll = MightexDLLWrapper(dll_path)
        self._dev_id = dev_id

        self.add_parameter('exposure_time',
                           unit='s',
                           set_cmd=self.set_exp_t,
                           get_cmd=self.get_exp_t,
                           get_parser=float,
                           vals=qc.validators.Numbers(1e-4, 6.5))

        self.add_parameter('device_on',
                           set_cmd=self.set_onoff,
                           get_cmd=self.is_on,
                           vals=qc.validators.Bool())

        self.add_parameter('serial',
                           get_cmd=self.get_serialno)

        self.add_parameter('auto_drk_on',
                           set_cmd=self.set_auto_drk,
                           get_cmd=self.is_auto_drk,
                           vals=qc.validators.Bool())

        self.add_parameter('wl_axis',
                           parameter_class=WavelengthAxis,
                           vals=vals.Arrays(shape=(MightexSSE_1304_U.PIXELS,)),
                           label='Wavelength',
                           unit='m')

        self.add_parameter('spectrum',
                           parameter_class=IntensitySpectrum,
                           setpoints=(self.wl_axis,),
                           vals=vals.Arrays(shape=(MightexSSE_1304_U.PIXELS,)),
                           label='Intensity')

        self.add_parameter('averaging',
                           get_cmd=self.get_averaging,
                           set_cmd=self.set_averaging,
                           vals=vals.Ints(1))

        self.add_parameter('darkframe',
                           get_cmd=self.rec_darkframe,
                           set_cmd=self.set_darkframe,
                           vals=vals.Arrays(shape=(MightexSSE_1304_U.PIXELS,)))

        self.connect()

        self.drkdata = np.zeros((MightexSSE_1304_U.PIXELS,))
        self.calib = None

    def connect(self):
        """ Connects to the spectrometer and initializes the dll. """
        self.free_device()  # there might be leftovers from previous runs

        n_connected = self._dll.init(None)
        if n_connected < self._dev_id:
            raise RuntimeError(f'No Mightex with id {self._dev_id} connected')

    def disconnect(self):
        """ Releases all resources for the spectrometer. """
        self._dll.uninit()

    def __del__(self):
        self.free_device()

    def set_exp_t(self, t: float) -> None:
        """ Sets the exposure time to `t` seconds. """
        # The spectrometer can only handle multiples of 100 microseconds
        self._exp_t = int(round(t * 1e4) * 100)
        self._dll.set_exposuretime(self._dev_id, self._exp_t)

    def get_exp_t(self) -> float:
        return self._exp_t / 1e6  # in seconds

    def set_onoff(self, on: bool) -> None:
        """ Turns the device on and off. """
        self._is_on = on
        self._dll.set_activestatus(self._dev_id, 1 if on else 0)

    def is_on(self) -> bool:
        return self._is_on

    def get_serialno(self) -> str:
        """ Retrieves the device's serial number. """
        mod_no = ct.create_string_buffer(b'\000' * 32)
        ser_no = ct.create_string_buffer(b'\000' * 32)
        self._dll.get_serialno(self._dev_id, mod_no, ser_no)
        return mod_no.value.decode() + ':' + ser_no.value.decode()

    def set_auto_drk(self, on: bool) -> None:
        """ Sets auto dark compensation for the spectrometer. """
        self._auto_drk_on = on
        self._dll.set_auto_dark(self._dev_id, 1, 1 if on else 0)

    def is_auto_drk(self) -> bool:
        return self._auto_drk_on

    def get_single_frame(self) -> np.ndarray:
        """ Retrieves a raw single frame from the spectrometer. """
        buffer = ct.POINTER(FrameRecord)()

        self._dll.start_framegrab(self._dev_id)
        self._dll.get_framedata(self._dev_id, 1, 1, ct.byref(buffer))

        return npct.as_array(buffer.contents.CCDData,
                             shape=(MightexSSE_1304_U.PIXELS,)).copy()

    def load_calibration(self, calib_file=None):
        """ Loads the calibration coefficients for the wavelength axis
        from `calib_file`. If `calib_file` is None, tries to load from
        the default file. """
        if calib_file is None:
            calib_file = Path(__file__).resolve().parent / 'calibration.json'

        try:
            calib_coeffs = json.load(calib_file)
        except AttributeError:
            # `calib_file` is not an opened file, try to open it
            with open(calib_file, 'r') as f:
                calib_coeffs = json.load(f)

        self.calib = np.poly1d(calib_coeffs)

    def get_averaging(self) -> int:
        return self._avg

    def set_averaging(self, avg: int):
        """ Sets the number of frames that should be averaged
        for each measurement. """
        self._avg = avg if avg > 0 else 1

    def rec_darkframe(self) -> np.ndarray:
        self.drkdata = self.get_single_frame()
        return self.drkdata

    def set_darkframe(self, drk: np.ndarray):
        self.drkdata = drk.reshape((MightexSSE_1304_U.PIXELS,))

