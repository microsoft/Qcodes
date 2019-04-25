""" This module communicates with the SSE SDK dll provided by Mightex. """

import ctypes as ct
import numpy.ctypeslib as npct


class FrameRecord(ct.Structure):
    """ This struct is used by the DLL to communicate recorded data. """
    _fields_ = [
            ('CCDData', ct.POINTER(ct.c_double)),
            ('CalibData', ct.POINTER(ct.c_double)),
            ('AbsIntData', ct.POINTER(ct.c_double))]


class MightexDLLWrapper(object):
    """ Provides the functions from the Mightex SDK dll. For documentation
    on the API see the manual provided with the SDK. """

    def __init__(self, dllpath=r'C:\windows\system32'):
        self._dll = npct.load_library('MT_Spectrometer_SDK.dll', dllpath)

        # map the dll functions to class methods

        self.init = self._dll.MTSSE_InitDevice
        self.init.argtypes = [ct.c_void_p]

        self.uninit = self._dll.MTSSE_UnInitDevice
        self.uninit.argtypes = []

        self.start_framegrab = self._dll.MTSSE_StartFrameGrab
        self.start_framegrab.argtypes = [ct.c_int]

        self.stop_framegrab = self._dll.MTSSE_StopFrameGrab
        self.stop_framegrab.argtypes = []

        self.get_framedata = self._dll.MTSSE_GetDeviceSpectrometerFrameData
        self.get_framedata.argtypes = [ct.c_int, ct.c_int, ct.c_int,
                                       ct.POINTER(ct.POINTER(FrameRecord))]

        self.get_serialno = self._dll.MTSSE_GetDeviceModuleNoSerialNo
        self.get_serialno.argtypes = [ct.c_int, ct.c_char_p,
                                      ct.c_char_p]

        self.set_exposuretime = self._dll.MTSSE_SetDeviceExposureTime
        self.set_exposuretime.argtypes = [ct.c_int, ct.c_int]

        self.set_avg_framenum = self._dll.MTSSE_SetDeviceAverageFrameNum
        self.set_avg_framenum.argtypes = [ct.c_int, ct.c_int]

        self.set_activestatus = self._dll.MTSSE_SetDeviceActiveStatus
        self.set_activestatus.argtypes = [ct.c_int, ct.c_int]

        self.set_auto_dark = self._dll.MTSSE_SetDeviceSpectrometerAutoDarkStatus
        self.set_auto_dark.argtypes = [ct.c_int, ct.c_int, ct.c_int]

