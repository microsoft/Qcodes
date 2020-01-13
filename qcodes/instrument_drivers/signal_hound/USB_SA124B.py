from time import sleep
import numpy as np
import ctypes as ct
import logging
from enum import IntEnum
from typing import Dict, Union, Optional, Any, Tuple

from qcodes.instrument.base import Instrument
import qcodes.utils.validators as vals
from qcodes.instrument.parameter import Parameter, ArrayParameter, \
    ParameterWithSetpoints

log = logging.getLogger(__name__)

number = Union[int, float]

class TraceParameter(Parameter):
    """
    A parameter that used a flag on the instrument to keeps track of if it's
    value has been synced to the instrument. It is intended that this
    type of parameter is synced using an external method which resets the flag.

    This is most likely used similar to a ``ManualParameter``
    I.e. calling set/get will not communicate with the instrument.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def set_raw(self, value: Any) -> None: # pylint: disable=method-hidden
        if not isinstance(self.instrument, SignalHound_USB_SA124B):
            raise RuntimeError("TraceParameter only works with "
                               "'SignalHound_USB_SA124B'")
        self.instrument._parameters_synced = False


class ExternalRefParameter(TraceParameter):
    """
    Parameter that handles the fact that external reference can only be
    enabled but not disabled.

    From the manual:

    Once a device has successfully switched to an external reference it
    must remain using it until the device is closed, and it is undefined
    behavior to disconnect the reference input from the reference BNC port.
    """

    def set_raw(self, value: bool) -> None:  # pylint: disable=method-hidden
        if self.get_latest() is True and value is False:
            raise RuntimeError("Signal Hound does not support disabling "
                               "external reference. To switch back to internal "
                               "reference close the device and start again.")
        super().set_raw(value)


class ScaleParameter(TraceParameter):
    """
    Parameter that handels changing the unit when the scale is changed.
    """

    def set_raw(self, value: bool) -> None:  # pylint: disable=method-hidden
        if not isinstance(self.instrument, SignalHound_USB_SA124B):
            raise RuntimeError("ScaleParameter only works with "
                               "'SignalHound_USB_SA124B'")
        if value in ('log-scale', 'log-full-scale'):
            unit = 'dBm'
        elif value in ('lin-scale', 'lin-full-scale'):
            unit = 'mV'
        else:
            raise RuntimeError("Unsupported scale")
        self.instrument.trace.unit = unit
        self.instrument.power.unit = unit
        super().set_raw(value)


class SweepTraceParameter(TraceParameter):
    """
    An extension to TraceParameter that keeps track of the trace setpoints in
    addition to the functionality of `TraceParameter`
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def set_raw(self, value: Any) -> None:  # pylint: disable=method-hidden
        if not isinstance(self.instrument, SignalHound_USB_SA124B):
            raise RuntimeError("SweepTraceParameter only works with "
                               "'SignalHound_USB_SA124B'")
        self.instrument._trace_updated = False
        super().set_raw(value)


class FrequencySweep(ArrayParameter):
    """
    Hardware controlled parameter class for SignalHound_USB_SA124B.

    Instrument returns an array of powers for different frequencies

    Args:
        name: parameter name
        instrument: instrument the parameter belongs to
        sweep_len: Number of steps in sweep
        start_freq: Starting frequency
        stepsize: Size of a frequency step

    Methods:
          set_sweep(sweep_len, start_freq, stepsize): sets the shapes and
              setpoint arrays of the parameter to correspond with the sweep
          get(): executes a sweep and returns magnitude and phase arrays

    """
    def __init__(self, name: str, instrument: 'SignalHound_USB_SA124B',
                 sweep_len: int, start_freq: number, stepsize: number) -> None:
        super().__init__(name, shape=(sweep_len,),
                         instrument=instrument,
                         unit='dBm',
                         label='Magnitude',
                         setpoint_units=('Hz',),
                         setpoint_labels=(f'Frequency',),
                         setpoint_names=(f'frequency',))
        self.set_sweep(sweep_len, start_freq, stepsize)

    def set_sweep(self, sweep_len: int, start_freq: number,
                  stepsize: number) -> None:
        """
        Set the setpoints of the Array parameter representing a frequency
        sweep.

        Args:
            sweep_len: Number of points in the sweep
            start_freq: Starting frequency of the sweep
            stepsize: Size of step between individual points

        """
        if not isinstance(self.instrument, SignalHound_USB_SA124B):
            raise RuntimeError("'FrequencySweep' is only implemented"
                               "for 'SignalHound_USB_SA124B'")
        end_freq = start_freq + stepsize*(sweep_len-1)
        freq_points = tuple(np.linspace(start_freq, end_freq, sweep_len))
        self.setpoints = (freq_points,)
        self.shape = (sweep_len,)
        self.instrument._trace_updated = True

    def get_raw(self) -> np.ndarray:
        if self.instrument is None:
            raise RuntimeError("No instrument is attached to"
                               "'FrequencySweep'")
        if not isinstance(self.instrument, SignalHound_USB_SA124B):
            raise RuntimeError("'FrequencySweep' is only implemented"
                               "for 'SignalHound_USB_SA124B'")
        if not self.instrument._trace_updated:
            raise RuntimeError('trace not updated, run configure to update')
        data = self.instrument._get_sweep_data()
        sleep(2*self.instrument.sleep_time.get())
        return data


class SignalHound_USB_SA124B(Instrument):
    """
    QCoDeS driver for the SignalHound USB SA124B

    The driver needs Signal Hounds software
    `Spike <https://signalhound.com/spike/>`_ installed to function.
    In addition you may need to install Microsoft Visual Studio C++
    Redistributable for the driver to function within QCoDeS.
    At the time of writing the current version of Spike (3.2.3) uses
    `Microsoft Visual Studio C++ Redistributable 2012
    <https://www.microsoft.com/en-us/download/details.aspx?id=30679>`_

    """
    dll_path = 'C:\\Program Files\\Signal Hound\\Spike\\sa_api.dll'

    def __init__(self, name, dll_path=None, **kwargs):
        """
        Args:
            name: Name of the instrument.
            dll_path: Path to ``sa_api.dll`` Defaults to the default dll within
                Spike installation
            **kwargs:
        """
        super().__init__(name, **kwargs)
        self._parameters_synced = False
        self._trace_updated = False
        log.info('Initializing instrument SignalHound USB 124B')
        self.dll = ct.CDLL(dll_path or self.dll_path)

        self._set_ctypes_argtypes()

        self.hf = Constants
        self.add_parameter('frequency',
                           label='Frequency',
                           unit='Hz',
                           initial_value=5e9,
                           vals=vals.Numbers(),
                           parameter_class=SweepTraceParameter,
                           docstring='Center frequency for sweep.'
                                     'This is the set center, the actual '
                                     'center may be subject to round off '
                                     'compared to this value')
        self.add_parameter('span',
                           label='Span',
                           unit='Hz',
                           initial_value=.25e6,
                           vals=vals.Numbers(),
                           parameter_class=SweepTraceParameter,
                           docstring='Width of frequency span'
                                     'This is the set span, the actual '
                                     'span may be subject to round off '
                                     'compared to this value'
                           )
        self.add_parameter('npts',
                           label='Number of Points',
                           get_cmd=self._get_npts,
                           set_cmd=False,
                           docstring='Number of points in frequency sweep.')
        self.add_parameter('avg',
                           label='Averages',
                           initial_value=1,
                           get_cmd=None,
                           set_cmd=None,
                           vals=vals.Ints(),
                           docstring='Number of averages to perform. '
                                     'Averages are performed in software by '
                                     'acquiring multiple sweeps')
        self.add_parameter('ref_lvl',
                           label='Reference power',
                           unit='dBm',
                           initial_value=0,
                           vals=vals.Numbers(max_value=20),
                           parameter_class=TraceParameter,
                           docstring="Setting reference level will "
                                     "automatically select gain and attenuation"
                                     "optimal for measuring at and below "
                                     "this level")
        self.add_parameter('external_reference',
                           initial_value=False,
                           vals=vals.Bool(),
                           parameter_class=ExternalRefParameter,
                           docstring='Use an external 10 MHz reference source. '
                                     'Note that Signal Hound does not support '
                                     'disabling external ref. To disable close '
                                     'the connection and restart.')
        self.add_parameter('device_type',
                           set_cmd=False,
                           get_cmd=self._get_device_type)
        self.add_parameter('device_mode',
                           get_cmd=lambda: 'sweeping',
                           set_cmd=False,
                           docstring='The driver currently only  '
                                     'supports sweeping mode. '
                                     'It is therefor not possible'
                                     'to set this parameter to anything else')
        self.add_parameter('acquisition_mode',
                           get_cmd=lambda: 'average',
                           set_cmd=False,
                           docstring="The driver only supports averaging "
                                     "mode it is therefor not possible to set"
                                     "this parameter to anything else")
        self.add_parameter('rbw',
                           label='Resolution Bandwidth',
                           unit='Hz',
                           initial_value=1e3,
                           vals=vals.Numbers(0.1, 250e3),
                           parameter_class=TraceParameter,
                           docstring='Resolution Bandwidth (RBW) is'
                                     'the bandwidth of '
                                     'spectral energy represented in each '
                                     'frequency bin')
        self.add_parameter('vbw',
                           label='Video Bandwidth',
                           unit='Hz',
                           initial_value=1e3,
                           vals=vals.Numbers(),
                           parameter_class=TraceParameter,
                           docstring='The video bandwidth (VBW) is applied '
                                     'after the signal has been converted to '
                                     'frequency domain as power, voltage, '
                                     'or log units. It is implemented as a '
                                     'simple rectangular window, averaging the '
                                     'amplitude readings for each frequency '
                                     'bin over several overlapping FFTs. '
                                     'For best performance use RBW as the VBW.')

        self.add_parameter('reject_image',
                           label='Reject image',
                           unit='',
                           initial_value=True,
                           parameter_class=TraceParameter,
                           get_cmd=None,
                           docstring="Apply software filter to remove "
                                     "undersampling mirroring",
                           vals=vals.Bool())
        self.add_parameter('sleep_time',
                           label='Sleep time',
                           unit='s',
                           initial_value=0.1,
                           get_cmd=None,
                           set_cmd=None,
                           docstring="Time to sleep before and after "
                                     "getting data from the instrument",
                           vals=vals.Numbers(0))
        # We don't know the correct values of
        # the sweep parameters yet so we supply
        # some defaults. The correct will be set when we call configure below
        self.add_parameter(name='trace',
                           sweep_len=1,
                           start_freq=1,
                           stepsize=1,
                           parameter_class=FrequencySweep)
        self.add_parameter('power',
                           label='Power',
                           unit='dBm',
                           get_cmd=self._get_power_at_freq,
                           set_cmd=False,
                           docstring="The maximum power in a window of 250 kHz"
                                     "around the specified  frequency with "
                                     "Resolution bandwidth set to 1 kHz."
                                     "The integration window is specified by "
                                     "the VideoBandWidth (set by vbw)")
        # scale is defined after the trace and power parameter so that
        # it can change the units of those in it's set method when the
        # scale changes
        self.add_parameter('scale',
                           initial_value='log-scale',
                           vals=vals.Enum('log-scale', 'lin-scale',
                                          'log-full-scale', 'lin-full-scale'),
                           parameter_class=ScaleParameter)

        self.add_parameter('frequency_axis',
                           label='Frequency',
                           unit='Hz',
                           get_cmd=self._get_freq_axis,
                           set_cmd=False,
                           vals=vals.Arrays(shape=(self.npts,)),
                           snapshot_value=False
                           )
        self.add_parameter('freq_sweep',
                           label='Power',
                           unit='depends on mode',
                           get_cmd=self._get_sweep_data,
                           set_cmd=False,
                           parameter_class=ParameterWithSetpoints,
                           vals=vals.Arrays(shape=(self.npts,)),
                           setpoints=(self.frequency_axis,),
                           snapshot_value=False)

        self.openDevice()
        self.configure()

        self.connect_message()

    def _set_ctypes_argtypes(self) -> None:
        """
        Set the expected argtypes for function calls in the sa_api dll
        These should match the function signatures defined in the sa-api
        header files included with the signal hound sdk
        """
        self.dll.saConfigCenterSpan.argtypes = [ct.c_int,
                                                ct.c_double,
                                                ct.c_double]
        self.dll.saConfigAcquisition.argtypes = [ct.c_int,
                                                 ct.c_int,
                                                 ct.c_int]
        self.dll.saConfigLevel.argtypes = [ct.c_int,
                                           ct.c_double]
        self.dll.saSetTimebase.argtypes = [ct.c_int,
                                           ct.c_int]
        self.dll.saConfigSweepCoupling.argypes = [ct.c_int,
                                                  ct.c_double,
                                                  ct.c_double,
                                                  ct.c_bool]
        self.dll.saInitiate.argtypes = [ct.c_int,
                                        ct.c_int,
                                        ct.c_int]
        self.dll.saOpenDevice.argtypes = [ct.POINTER(ct.c_int)]
        self.dll.saCloseDevice.argtypes = [ct.c_int]
        self.dll.saPreset.argtypes = [ct.c_int]
        self.dll.saGetDeviceType.argtypes = [ct.c_int,
                                             ct.POINTER(ct.c_int)]
        self.dll.saQuerySweepInfo.argtypes = [ct.c_int,
                                              ct.POINTER(ct.c_int),
                                              ct.POINTER(ct.c_double),
                                              ct.POINTER(ct.c_double)]
        self.dll.saGetSweep_32f.argtypes = [ct.c_int, ct.POINTER(ct.c_float),
                                            ct.POINTER(ct.c_float)]
        self.dll.saGetSerialNumber.argtypes = [ct.c_int,
                                               ct.POINTER(ct.c_int)]
        self.dll.saGetFirmwareString.argtypes = [ct.c_int,
                                                 ct.c_char_p]

    def _get_npts(self) -> int:
        if not self._parameters_synced:
            self.sync_parameters()
        sweep_info = self.QuerySweep()
        sweep_len = sweep_info[0]
        return sweep_len

    def _update_trace(self) -> None:
        """
        Private method to sync changes of the
        frequency axis to the setpoints of the
        trace parameter. This also set the units
        of power and trace.
        """
        sweep_info = self.QuerySweep()
        self.npts.cache.set(sweep_info[0])
        self.trace.set_sweep(*sweep_info)

    def sync_parameters(self) -> None:
        """
        Sync parameters sets the configuration of the instrument using the
        parameters specified in the Qcodes instrument.

        Sync parameters consists of five parts
            1. Center span configuration (freqs and span)
            2. Acquisition configuration
                lin-scale/log-scale
                avg/max power
            3. Configuring the external 10MHz reference
            4. Configuration of the mode that is being used
            5. Acquisition mode. At the moment only `sweeping` is implemented

        This does not currently implement Configuration of the tracking
        generator used in VNA mode
        """

        # 1. CenterSpan Configuration
        center = ct.c_double(self.frequency())
        span = ct.c_double(self.span())
        log.info('Setting device CenterSpan configuration.')

        err = self.dll.saConfigCenterSpan(self.deviceHandle, center, span)
        self.check_for_error(err, 'saConfigCenterSpan')

        # 2. Acquisition configuration
        detectorVals = {
            'min-max': ct.c_int(self.hf.sa_MIN_MAX),
            'average': ct.c_int(self.hf.sa_AVERAGE)
        }
        scaleVals = {
            'log-scale': ct.c_int(self.hf.sa_LOG_SCALE),
            'lin-scale': ct.c_int(self.hf.sa_LIN_SCALE),
            'log-full-scale': ct.c_int(self.hf.sa_LOG_FULL_SCALE),
            'lin-full-scale': ct.c_int(self.hf.sa_LIN_FULL_SCALE)
        }
        detector = detectorVals[self.acquisition_mode()]
        scale = scaleVals[self.scale()]

        err = self.dll.saConfigAcquisition(self.deviceHandle, detector, scale)
        self.check_for_error(err, 'saConfigAcquisition')

        # 3. Reference Level configuration
        log.info('Setting device reference level configuration.')
        err = self.dll.saConfigLevel(
            self.deviceHandle, ct.c_double(self.ref_lvl()))
        self.check_for_error(err, 'saConfigLevel')

        # 4. External Reference configuration
        if self.external_reference():
            external = self.hf.sa_REF_EXTERNAL_IN
            log.info('Setting reference frequency from external source.')
            err = self.dll.saSetTimebase(self.deviceHandle,
                                         external)
            self.check_for_error(err, 'saSetTimebase')


        reject_var = ct.c_bool(self.reject_image())
        log.info('Setting device Sweeping configuration.')
        err = self.dll.saConfigSweepCoupling(
            self.deviceHandle, ct.c_double(self.rbw()),
            ct.c_double(self.vbw()), reject_var)
        self.check_for_error(err, 'saConfigSweepCoupling')


        modeOpts = {
            'sweeping': self.hf.sa_SWEEPING,
            'real_time': self.hf.sa_REAL_TIME,  # not implemented
            'IQ': self.hf.sa_IQ,  # not implemented
            'idle': self.hf.sa_IDLE
        }
        mode = modeOpts[self.device_mode()]
        # the third argument to saInitiate is a flag that is
        # currently not used
        err = self.dll.saInitiate(self.deviceHandle, mode, 0)
        extrainfo: Optional[str] = None
        if err == saStatus.saInvalidParameterErr:
            extrainfo = """
                 In real-time mode, this value may be returned if the span
                 limits defined in the API header are broken. Also in
                 real-time mode, this error will be returned if the
                 resolution bandwidth is outside the limits defined in
                 the API header.
                 In time-gate analysis mode this error will be returned if
                 span limits defined in the API header are broken. Also in
                 time gate analysis, this error is returned if the
                 bandwidth provided require more samples for processing
                 than is allowed in the gate length. To fix this
                 increase rbw/vbw.
             """
        elif err == saStatus.saBandwidthErr:
            extrainfo = 'RBW is larger than your span. (Sweep Mode)!'
        self.check_for_error(err, 'saInitiate', extrainfo)

        self._parameters_synced = True

    def configure(self) -> None:
        """
        Syncs parameters to the Instrument and updates the setpoint of the
        trace.
        """
        self.sync_parameters()
        self._update_trace()

    def openDevice(self) -> None:
        """
        Opens connection to the instrument
        """
        log.info('Opening Device')
        self.deviceHandle = ct.c_int(0)
        deviceHandlePnt = ct.pointer(self.deviceHandle)
        err = self.dll.saOpenDevice(deviceHandlePnt)
        self.check_for_error(err, 'saOpenDevice')
        self.device_type()

    def close(self) -> None:
        """
        Close connection to the instrument.
        """
        log.info('Closing Device with handle num: '
                 f'{self.deviceHandle.value}')

        try:
            self.abort()
            log.info('Running acquistion aborted.')
        except Exception as e:
            # it's ok to catch any exception here
            # as we are tearing down the instrument we might
            # as well try to continue
            log.warning(f'Could not abort acquisition: {e}')

        err = self.dll.saCloseDevice(self.deviceHandle)
        self.check_for_error(err, 'saCloseDevice')
        log.info(f'Closed Device with handle num: {self.deviceHandle.value}')
        super().close()

    def abort(self) -> None:
        """
        Abort any running acquisition.
        """
        log.info('Stopping acquisition')

        err = self.dll.saAbort(self.deviceHandle)
        extrainfo: Optional[str] = None
        if err == saStatus.saDeviceNotConfiguredErr:
            extrainfo = 'Device was already idle! Did you call abort ' \
                        'without ever calling initiate()'

        self.check_for_error(err, 'saAbort', extrainfo)

    def preset(self) -> None:
        """
        Like close but performs a hardware reset before closing the
        connection.
        """
        log.warning('Performing hardware-reset of device!')

        err = self.dll.saPreset(self.deviceHandle)
        self.check_for_error(err, 'saPreset')
        super().close()

    def _get_device_type(self) -> str:
        """
        Returns the model string of the Spectrum Analyzer.
        """
        log.info('Querying device for model information')

        devType = ct.c_int32(0)
        devTypePnt = ct.pointer(devType)

        err = self.dll.saGetDeviceType(self.deviceHandle, devTypePnt)
        self.check_for_error(err, 'saGetDeviceType')

        if devType.value == self.hf.saDeviceTypeNone:
            dev = 'No device'
        elif devType.value == self.hf.saDeviceTypeSA44:
            dev = 'sa44'
        elif devType.value == self.hf.saDeviceTypeSA44B:
            dev = 'sa44B'
        elif devType.value == self.hf.saDeviceTypeSA124A:
            dev = 'sa124A'
        elif devType.value == self.hf.saDeviceTypeSA124B:
            dev = 'sa124B'
        else:
            raise ValueError('Unknown device type!')
        return dev

    ########################################################################

    def QuerySweep(self) -> Tuple[int, float, float]:
        """
        Queries the sweep for information on the parameters that defines the
            x axis of the sweep

        Returns:
            number of points in sweep, start frequency and step size
        """

        sweep_len = ct.c_int(0)
        start_freq = ct.c_double(0)
        stepsize = ct.c_double(0)
        err = self.dll.saQuerySweepInfo(self.deviceHandle,
                                        ct.pointer(sweep_len),
                                        ct.pointer(start_freq),
                                        ct.pointer(stepsize))
        self.check_for_error(err, 'saQuerySweepInfo')
        return sweep_len.value, start_freq.value, stepsize.value

    def _get_sweep_data(self) -> np.ndarray:
        """
        This function performs a sweep over the configured ranges.
        The result of the sweep is returned along with the sweep points

        returns:
            datamin numpy array
        """
        if not self._parameters_synced:
            self.sync_parameters()
        sweep_len, _, _ = self.QuerySweep()


        data = np.zeros(sweep_len)
        Navg = self.avg()
        for i in range(Navg):

            datamin = np.zeros((sweep_len), dtype=np.float32)
            datamax = np.zeros((sweep_len), dtype=np.float32)

            minarr = datamin.ctypes.data_as(ct.POINTER(ct.c_float))
            maxarr = datamax.ctypes.data_as(ct.POINTER(ct.c_float))

            sleep(self.sleep_time.get())  # Added extra sleep for updating issue
            err = self.dll.saGetSweep_32f(self.deviceHandle, minarr, maxarr)
            self.check_for_error(err, 'saGetSweep_32f')
            data += datamin

        return data / Navg

    def _get_power_at_freq(self) -> float:
        """
        Returns the maximum power in a window of 250 kHz
        around the specified  frequency with Resolution bandwidth set to 1 kHz.
        The integration window is specified by the VideoBandWidth (set by vbw)
        """
        original_span = self.span()
        original_rbw = self.rbw()
        needs_reset = False
        if not (original_span == 0.25e6 and original_rbw == 1e3):
            needs_reset = True
            self.span(0.25e6)
            self.rbw(1e3)
        if not self._parameters_synced:
            # call configure to update both
            # the parameters on the device and the
            # setpoints and units
            self.configure()
        data = self._get_sweep_data()
        max_power = np.max(data)
        if needs_reset:
            self.span(original_span)
            self.rbw(original_rbw)
            self.configure()
        sleep(2*self.sleep_time.get())
        return max_power

    @staticmethod
    def check_for_error(err: int, source: str, extrainfo: str=None) -> None:
        if err != saStatus.saNoError:
            err_str = saStatus(err).name
            if err > 0:
                msg = (f'During call of {source} the following'
                       f'Warning: {err_str} was raised')
                if extrainfo is not None:
                    msg = msg + f'\n Extra info: {extrainfo}'
                log.warning(msg)
            else:
                msg = (f'During call of {source} the following Error: '
                       f'{err_str} was raised')
                if extrainfo is not None:
                    msg = msg + f'\n Extra info: {extrainfo}'
                raise IOError(msg)
        else:
            msg = 'Call to {source} was successful'
            if extrainfo is not None:
                msg = msg + f'\n Extra info: {extrainfo}'
            log.info(msg)

    def get_idn(self) -> Dict[str, Optional[str]]:
        output: Dict[str, Optional[str]] = {}
        output['vendor'] = 'Signal Hound'
        output['model'] = self._get_device_type()
        serialnumber = ct.c_int32()
        err = self.dll.saGetSerialNumber(self.deviceHandle,
                                         ct.pointer(serialnumber))
        self.check_for_error(err, 'saGetSerialNumber')
        output['serial'] = str(serialnumber.value)
        fw_version = (ct.c_char*17)()
        # the manual says that this must be at least 16 char
        # but not clear if that includes a termination zero so
        # make it 17 just in case
        err = self.dll.saGetFirmwareString(self.deviceHandle, fw_version)
        self.check_for_error(err, 'saGetFirmwareString')
        output['firmware'] = fw_version.value.decode('ascii')
        return output

    def _get_freq_axis(self) -> np.ndarray:
        if not self._parameters_synced:
            self.sync_parameters()
        sweep_len, start_freq, stepsize = self.QuerySweep()
        end_freq = start_freq + stepsize*(sweep_len-1)
        freq_points = np.linspace(start_freq, end_freq, sweep_len)
        return freq_points


class Constants:
    """
    These constants are defined in sa_api.h as part of the the Signal Hound
    SDK
    """
    SA_MAX_DEVICES = 8

    saDeviceTypeNone = 0
    saDeviceTypeSA44 = 1
    saDeviceTypeSA44B = 2
    saDeviceTypeSA124A = 3
    saDeviceTypeSA124B = 4

    sa44_MIN_FREQ = 1.0
    sa124_MIN_FREQ = 100.0e3
    sa44_MAX_FREQ = 4.4e9
    sa124_MAX_FREQ = 13.0e9
    sa_MIN_SPAN = 1.0
    sa_MAX_REF = 20
    sa_MAX_ATTEN = 3
    sa_MAX_GAIN = 2
    sa_MIN_RBW = 0.1
    sa_MAX_RBW = 6.0e6
    sa_MIN_RT_RBW = 100.0
    sa_MAX_RT_RBW = 10000.0
    sa_MIN_IQ_BANDWIDTH = 100.0
    sa_MAX_IQ_DECIMATION = 128

    sa_IQ_SAMPLE_RATE = 486111.111

    sa_IDLE = -1
    sa_SWEEPING = 0x0
    sa_REAL_TIME = 0x1
    sa_IQ = 0x2
    sa_AUDIO = 0x3
    sa_TG_SWEEP = 0x4

    sa_MIN_MAX = 0x0
    sa_AVERAGE = 0x1

    sa_LOG_SCALE = 0x0
    sa_LIN_SCALE = 0x1
    sa_LOG_FULL_SCALE = 0x2
    sa_LIN_FULL_SCALE = 0x3

    sa_AUTO_ATTEN = -1
    sa_AUTO_GAIN = -1

    sa_LOG_UNITS = 0x0
    sa_VOLT_UNITS = 0x1
    sa_POWER_UNITS = 0x2
    sa_BYPASS = 0x3

    sa_AUDIO_AM = 0x0
    sa_AUDIO_FM = 0x1
    sa_AUDIO_USB = 0x2
    sa_AUDIO_LSB = 0x3
    sa_AUDIO_CW = 0x4

    TG_THRU_0DB = 0x1
    TG_THRU_20DB = 0x2

    sa_REF_UNUSED = 0
    sa_REF_INTERNAL_OUT = 1
    sa_REF_EXTERNAL_IN = 2


class saStatus(IntEnum):
    saUnknownErr = -666
    saFrequencyRangeErr = 99
    saInvalidDetectorErr = -95
    saInvalidScaleErr = -94
    saBandwidthErr = -91
    saExternalReferenceNotFound = -89
    # Device specific errors
    saOvenColdErr = -20
    # Data errors
    saInternetErr = -12
    saUSBCommErr = -11
    # General configuration errors
    saTrackingGeneratorNotFound = -10
    saDeviceNotIdleErr = -9
    saDeviceNotFoundErr = -8
    saInvalidModeErr = -7
    saNotConfiguredErr = -6
    saDeviceNotConfiguredErr = -6 # Added because key error raised
    saTooManyDevicesErr = -5
    saInvalidParameterErr = -4
    saDeviceNotOpenErr = -3
    saInvalidDeviceErr = -2
    saNullPtrErr = -1
    # No error
    saNoError = 0
    # Warnings
    saNoCorrections = 1
    saCompressionWarning = 2
    saParameterClamped = 3
    saBandwidthClamped = 4
