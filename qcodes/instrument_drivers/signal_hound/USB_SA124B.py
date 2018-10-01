from time import sleep
import numpy as np
import ctypes as ct
import logging
from enum import IntEnum
from typing import Dict, Union, Optional, Any, Tuple

from qcodes import Instrument, ArrayParameter, Parameter, validators as vals

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_raw(self, value: Any):  # pylint: disable=method-hidden
        if not isinstance(self.instrument, SignalHound_USB_SA124B):
            raise RuntimeError("TraceParameter only works with "
                               "'SignalHound_USB_SA124B'")
        self.instrument._parameters_synced = False
        self._save_val(value, validate=False)


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
                         unit='dB',
                         label='{} Magnitude'.format(
                             instrument.short_name),
                         setpoint_units=('Hz',),
                         setpoint_labels=(f'{instrument.short_name} '
                                          f'Frequency',),
                         setpoint_names=(f'{instrument.short_name}'
                                         f'_frequency',))
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
        if not isinstance(self.instrument, SignalHound_USB_SA124B):
            raise RuntimeError("'FrequencySweep' is only implemented"
                               "for 'SignalHound_USB_SA124B'")
        if not self.instrument._trace_updated:
            raise RuntimeError('trace not updated, run configure to update')
        data = self._instrument._get_averaged_sweep_data()
        sleep(0.2)
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
        self.hf = Constants

        self.add_parameter('frequency',
                           label='Frequency ',
                           unit='Hz',
                           initial_value=5e9,
                           vals=vals.Numbers(),
                           parameter_class=SweepTraceParameter)
        self.add_parameter('span',
                           label='Span ',
                           unit='Hz',
                           initial_value=.25e6,
                           vals=vals.Numbers(),
                           parameter_class=SweepTraceParameter)
        self.add_parameter('npts',
                           label='Number of Points',
                           get_cmd=None,
                           set_cmd=False)
        self.add_parameter('avg',
                           label='Averages',
                           initial_value=1,
                           get_cmd=None,
                           set_cmd=None,
                           vals=vals.Ints())
        self.add_parameter('power',
                           label='Power ',
                           unit='dBm',
                           get_cmd=self._get_power_at_freq,
                           set_cmd=False)
        self.add_parameter('ref_lvl',
                           label='Reference power ',
                           unit='dBm',
                           initial_value=0,
                           vals=vals.Numbers(max_value=20),
                           parameter_class=TraceParameter)
        self.add_parameter('external_reference',
                           initial_value=False,
                           vals=vals.Bool(),
                           parameter_class=TraceParameter)
        self.add_parameter('device_type',
                           set_cmd=False,
                           get_cmd=self._get_device_type)
        self.add_parameter('device_mode',
                           get_cmd=lambda: 'sweeping',
                           set_cmd=False)
        self.add_parameter('acquisition_mode',
                           get_cmd=lambda: 'average',
                           set_cmd=False)
        self.add_parameter('scale',
                           initial_value='log-scale',
                           vals=vals.Enum('log-scale', 'lin-scale',
                                          'log-full-scale', 'lin-full-scale'),
                           parameter_class=TraceParameter)
        self.add_parameter('running',
                           get_cmd=None,
                           set_cmd=None,
                           initial_value=False,
                           vals=vals.Bool())
        # rbw Resolution bandwidth in Hz. RBW can be arbitrary.
        self.add_parameter('rbw',
                           label='Resolution Bandwidth',
                           unit='Hz',
                           initial_value=1e3,
                           vals=vals.Numbers(),
                           parameter_class=TraceParameter)
        # vbw Video bandwidth in Hz. VBW must be less than or equal to RBW.
        # VBW can be arbitrary. For best performance use RBW as the VBW.
        # what does this even do though? nataliejpg
        self.add_parameter('vbw',
                           label='Video Bandwidth',
                           unit='Hz',
                           initial_value=1e3,
                           vals=vals.Numbers(),
                           parameter_class=TraceParameter)

        self.add_parameter('reject_image',
                           label='Reject image',
                           unit='',
                           initial_value=True,
                           parameter_class=TraceParameter,
                           get_cmd=None,
                           docstring="Apply software filter to remove "
                                     "undersampling mirroring",
                           vals=vals.Bool())

        self.openDevice()
        self.device_type()

        self._prepare_measurement()
        sweep_len, start_freq, stepsize = self.QuerySweep()
        self.add_parameter(name='trace',
                           sweep_len=sweep_len,
                           start_freq=start_freq,
                           stepsize=stepsize,
                           parameter_class=FrequencySweep)
        self.npts._save_val(sweep_len)

        self.connect_message()

    def _update_trace(self) -> None:
        """
        Private method to sync changes of the
        frequency axis to the setpoints of the
        trace parameter.
        """
        sweep_info = self.QuerySweep()
        self.npts._save_val(sweep_info[0])
        self.trace.set_sweep(*sweep_info)

    def _sync_parameters(self) -> None:
        """
        Sync parameters consists of five parts
            1. Center span configuration (freqs and span)
            2. Acquisition configuration
                lin-scale/log-scale
                avg/max power
            3. Configuring the external 10MHz reference
            4. Configuration of the mode that is being used
            5. Configuration of the tracking generator (not implemented)
                used in VNA mode

        Sync parameters sets the configuration of the instrument using the
        parameters specified in the Qcodes instrument.
        """

        # 1. CenterSpan Configuration
        center = ct.c_double(self.frequency())
        span = ct.c_double(self.span())
        log.info('Setting device CenterSpan configuration.')

        err = self.dll.saConfigCenterSpan(self.deviceHandle, center, span)
        self.check_for_error(err)

        # 2. Acquisition configuration
        detectorVals = {
            'min-max': ct.c_uint(self.hf.sa_MIN_MAX),
            'average': ct.c_uint(self.hf.sa_AVERAGE)
        }
        scaleVals = {
            'log-scale': ct.c_uint(self.hf.sa_LOG_SCALE),
            'lin-scale': ct.c_uint(self.hf.sa_LIN_SCALE),
            'log-full-scale': ct.c_uint(self.hf.sa_LOG_FULL_SCALE),
            'lin-full-scale': ct.c_uint(self.hf.sa_LIN_FULL_SCALE)
        }
        detector = detectorVals[self.acquisition_mode()]
        scale = scaleVals[self.scale()]

        err = self.dll.saConfigAcquisition(self.deviceHandle, detector, scale)
        self.check_for_error(err)

        # 3. Reference Level configuration
        log.info('Setting device reference level configuration.')
        err = self.dll.saConfigLevel(
            self.deviceHandle, ct.c_double(self.ref_lvl()))
        self.check_for_error(err)

        # 4. External Reference configuration
        if self.external_reference():
            log.info('Setting reference frequency from external source.')
            err = self.dll.saEnableExternalReference(self.deviceHandle)
            self.check_for_error(err)

        reject_var = ct.c_bool(self.reject_image())
        log.info('Setting device Sweeping configuration.')
        err = self.dll.saConfigSweepCoupling(
            self.deviceHandle, ct.c_double(self.rbw()),
            ct.c_double(self.vbw()), reject_var)
        self.check_for_error(err)
        self._parameters_synced = True

    def configure(self) -> None:
        self._prepare_measurement()
        self._update_trace()

    def openDevice(self) -> None:
        log.info('Opening Device')
        self.deviceHandle = ct.c_int(0)
        deviceHandlePnt = ct.pointer(self.deviceHandle)
        ret = self.dll.saOpenDevice(deviceHandlePnt)
        if ret != saStatus.saNoError:
            if ret == saStatus.saNullPtrErr:
                raise ValueError('Could not open device due to '
                                 'null-pointer error!')
            elif ret == saStatus.saDeviceNotOpenErr:
                raise ValueError('Could not open device!')
            else:
                raise ValueError('Could not open device due to unknown '
                                 'reason! Error = %d' % ret)

        self.devOpen = True
        self.device_type()

    def close(self) -> None:
        log.info('Closing Device with handle num: '
                 f'{self.deviceHandle.value}')

        try:
            self.dll.saAbort(self.deviceHandle)
            log.info('Running acquistion aborted.')
        except Exception as e:
            log.info(f'Could not abort acquisition: {e}')

        ret = self.dll.saCloseDevice(self.deviceHandle)
        if ret != saStatus.saNoError:
            raise ValueError('Error closing device: '
                             f'{saStatus(ret).name}')
        log.info(f'Closed Device with handle num: {self.deviceHandle.value}')
        self.devOpen = False
        self.running(False)
        super().close()

    def abort(self) -> None:
        """
        Abort any running acquisition.
        """
        log.info('Stopping acquisition')

        err = self.dll.saAbort(self.deviceHandle)
        if err == saStatus.saNoError:
            log.info('Call to abort succeeded.')
            self.running(False)
        elif err == saStatus.saDeviceNotOpenErr:
            raise IOError('Device not open!')
        elif err == saStatus.saDeviceNotConfiguredErr:
            raise IOError('Device was already idle! Did you call abort '
                          'without ever calling initiate()?')
        else:
            raise IOError(f'Unknown error setting abort! '
                          f'Error = {saStatus(err).name}')

    def preset(self) -> None:
        """
        Like close but performs a hardware reset before closing the
        connection
        """
        log.warning('Performing hardware-reset of device!')

        err = self.dll.saPreset(self.deviceHandle)
        if err == saStatus.saNoError:
            log.info('Call to preset succeeded.')
        elif err == saStatus.saDeviceNotOpenErr:
            raise IOError('Device not open!')
        else:
            raise IOError(f'Unknown error calling preset! '
                          f'Error = {saStatus(err).name}')
        super().close()

    def _get_device_type(self) -> str:
        """
        Returns the model string of the Spectrum Analyzer.
        """
        log.info('Querying device for model information')

        devType = ct.c_uint(0)
        devTypePnt = ct.pointer(devType)

        err = self.dll.saGetDeviceType(self.deviceHandle, devTypePnt)
        if err == saStatus.saNoError:
            pass
        elif err == saStatus.saDeviceNotOpenErr:
            raise IOError('Device not open!')
        elif err == saStatus.saNullPtrErr:
            raise IOError('Null pointer error!')
        else:
            raise IOError('Unknown error setting getDeviceType! '
                          f'Error = {saStatus(err).name}')

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

    def _initialise(self, flag: int=0) -> None:
        modeOpts = {
            'sweeping': self.hf.sa_SWEEPING,
            'real_time': self.hf.sa_REAL_TIME, # not implemented
            'IQ': self.hf.sa_IQ,  # not implemented
            'idle': self.hf.sa_IDLE
        }
        mode = modeOpts[self.device_mode()]
        err = self.dll.saInitiate(self.deviceHandle, mode, flag)

        ###################################
        # Below here only error handling
        ###################################
        if err == saStatus.saNoError:
            self.running(True)
            log.info('Call to initiate succeeded.')
        elif err == saStatus.saDeviceNotOpenErr:
            raise IOError('Device not open!')
        elif err == saStatus.saInvalidParameterErr:
            log.error(
                """
                saInvalidParameterErr!
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
            )
            raise IOError('The value for mode did not match any known value.')
        elif err == saStatus.saBandwidthErr:
            raise IOError('RBW is larger than your span. (Sweep Mode)!')
        self.check_for_error(err)

    def QuerySweep(self) -> Tuple[int, float, float]:
        """
        Queries the sweep for information on the parameters it uses
        """
        sweep_len = ct.c_int(0)
        start_freq = ct.c_double(0)
        stepsize = ct.c_double(0)
        err = self.dll.saQuerySweepInfo(self.deviceHandle,
                                        ct.pointer(sweep_len),
                                        ct.pointer(start_freq),
                                        ct.pointer(stepsize))
        if err == saStatus.saNoError:
            pass
        elif err == saStatus.saDeviceNotOpenErr:
            raise IOError('Device not open!')
        elif err == saStatus.saDeviceNotConfiguredErr:
            raise IOError('The device specified is not currently streaming!')
        elif err == saStatus.saNullPtrErr:
            raise IOError('Null pointer error!')
        else:
            raise IOError('Unknown error!')

        info = (sweep_len.value, start_freq.value, stepsize.value)
        return info

    def _get_sweep_data(self) -> np.ndarray:
        """
        This function performs a sweep over the configured ranges.
        The result of the sweep is returned along with the sweep points

        returns:
            datamin numpy array
        """
        if not self._parameters_synced:
            self._sync_parameters()
        sweep_len, _, _ = self.QuerySweep()

        minarr = (ct.c_float * sweep_len)()
        maxarr = (ct.c_float * sweep_len)()
        sleep(.1)  # Added extra sleep for updating issue
        err = self.dll.saGetSweep_32f(self.deviceHandle, minarr, maxarr)
        sleep(.1)  # Added extra sleep
        if not err == saStatus.saNoError:
            # if an error occurs tries preparing the device and then asks again
            print('Error raised in QuerySweepInfo, preparing for measurement')
            sleep(.1)
            self.prepare_for_measurement()
            sleep(.1)
            minarr = (ct.c_float * sweep_len)()
            maxarr = (ct.c_float * sweep_len)()
            err = self.dll.saGetSweep_32f(self.deviceHandle, minarr, maxarr)

        if err == saStatus.saNoError:
            pass
        elif err == saStatus.saDeviceNotOpenErr:
            raise IOError('Device not open!')
        elif err == saStatus.saDeviceNotConfiguredErr:
            raise IOError('The device specified is not currently streaming!')
        elif err == saStatus.saNullPtrErr:
            raise IOError('Null pointer error!')
        elif err == saStatus.saInvalidModeErr:
            raise IOError('Invalid mode error!')
        elif err == saStatus.saCompressionWarning:
            raise IOError('Input voltage overload!')
        elif err == saStatus.saUSBCommErr:
            raise IOError('Error occurred in the USB connection!')
        else:
            raise IOError('Unknown error!')

        datamin = np.array([minarr[elem] for elem in range(sweep_len)])

        return datamin

    def _get_averaged_sweep_data(self) -> np.ndarray:
        """
        Averages over SH.sweep Navg times

        """
        sweep_len, start_freq, stepsize = self.QuerySweep()
        data = np.zeros(sweep_len)
        Navg = self.avg()
        for i in range(Navg):
            data += self._get_sweep_data()
        return data / Navg

    def _get_power_at_freq(self) -> float:
        """
        Returns the maximum power in a window of 250kHz
        around the specified  frequency with Resolution bandwidth set to 1 KHz.
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
            self._prepare_measurement()
        data = self._get_averaged_sweep_data()
        max_power = np.max(data)
        if needs_reset:
            self.span(original_span)
            self.rbw(original_rbw)
            self._prepare_measurement()
        sleep(0.2)
        return max_power

    @staticmethod
    def check_for_error(err: int) -> None:
        if err != saStatus.saNoError:
            err_msg = saStatus(err).name
            if err > 0:
                print('Warning:', err_msg)
            else:
                raise IOError(err_msg)

    def _prepare_measurement(self) -> None:
        self._sync_parameters()
        self._initialise()

    def get_idn(self) -> Dict[str, Optional[str]]:
        output = {}
        output['vendor'] = 'Signal Hound'
        output['model'] = self._get_device_type()
        serialnumber = ct.c_int32()
        ret = self.dll.saGetSerialNumber(self.deviceHandle,
                                         ct.pointer(serialnumber))
        if ret != saStatus.saNoError:
            raise RuntimeError(f"Could not get serial number. "
                               f"Error was: {saStatus(ret).name}")
        output['serial'] = str(serialnumber.value)
        fw_version = (ct.c_char*17)()
        # the manual says that this must be at least 16 char
        # but not clear if that includes a termination zero so
        # make it 17 just in case
        ret = self.dll.saGetFirmwareString(self.deviceHandle, fw_version)
        if ret != saStatus.saNoError:
            raise RuntimeError(f"Could not get fw version. "
                               f"Error was: {saStatus(ret).name}")
        output['firmware'] = fw_version.value.decode('ascii')
        return output


class Constants:
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
