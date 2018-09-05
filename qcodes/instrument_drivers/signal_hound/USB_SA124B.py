from time import sleep, time
import numpy as np
import ctypes as ct
import logging
from functools import partial

from qcodes import Instrument, validators as vals

log = logging.getLogger(__name__)


class FrequencySweep(ArrayParameter):
    """
    Hardware controlled parameter class for SignalHound_USB_SA124B.

    Instrument returns an array of powers for different freuquenies

    Args:
        name: parameter name
        instrument: instrument the parameter belongs to
        start: starting frequency of sweep
        stop: ending frequency of sweep
        npts: number of points in frequency sweep

    Methods:
          set_sweep(start, stop, npts): sets the shapes and
              setpoint arrays of the parameter to correspond with the sweep
          get(): executes a sweep and returns magnitude and phase arrays

    """
    def __init__(self, name, instrument, center, span, npts):
        super().__init__(name, shape=(npts,),
                         instrument=instrument,
                         unit='dB',
                         label='{} magnitude'.format(
                             instrument.short_name),
                         setpoint_units=('Hz',),
                         setpoint_labels=('{} frequency'.format(instrument.short_name),),
                         setpoint_names=('{}_frequency'.format(instrument.short_name),))
        self.set_sweep(start, stop, npts)

    def set_sweep(self, sweep_len, start_freq, stepsize):
        end_freq = start_freq + stepsize*(sweep_len-1)
        freq_points = tuple(np.linspace(start_freq, end_freq, sweep_len))
        self.setpoints = (freq_points,)
        self.shape = (sweep_len,)

    def get_raw(self):
        data = self._instrument._get_averaged_sweep_data()
        return data


class SignalHound_USB_SA124B(Instrument):
    """
    QCoDeS driver for the SignalHound USB SA124B
    """
    dll_path = 'C:\Windows\System32\sa_api.dll'

    saStatus = {
        'saUnknownErr': -666,
        'saFrequencyRangeErr': -99,
        'saInvalidDetectorErr': -95,
        'saInvalidScaleErr': -94,
        'saBandwidthErr': -91,
        'saExternalReferenceNotFound': -89,
        # Device specific errors
        'saOvenColdErr': -20,
        # Data errors
        'saInternetErr': -12,
        'saUSBCommErr': -11,
        # General configuration errors
        'saTrackingGeneratorNotFound': -10,
        'saDeviceNotIdleErr': -9,
        'saDeviceNotFoundErr': -8,
        'saInvalidModeErr': -7,
        'saNotConfiguredErr': -6,
        'saDeviceNotConfiguredErr': -6,  # Added because key error raised
        'saTooManyDevicesErr': -5,
        'saInvalidParameterErr': -4,
        'saDeviceNotOpenErr': -3,
        'saInvalidDeviceErr': -2,
        'saNullPtrErr': -1,
        # No error
        'saNoError': 0,
        # Warnings
        'saNoCorrections': 1,
        'saCompressionWarning': 2,
        'saParameterClamped': 3,
        'saBandwidthClamped': 4
    }
    saStatus_inverted = dict((v, k) for k, v in saStatus.items())

    def __init__(self, name, dll_path=None, **kwargs):
        t0 = time()
        super().__init__(name, **kwargs)

        log.info('Initializing instrument SignalHound USB 124A')
        self.dll = ct.CDLL(dll_path or self.dll_path)
        self.hf = constants

        self.add_parameter('frequency',
                           label='Frequency ',
                           unit='Hz',
                           initial_value=5e9,
                           get_cmd=None,
                           set_cmd=self._set_frequency,
                           vals=vals.Numbers())
        self.add_parameter('span',
                           label='Span ',
                           unit='Hz',
                           initial_value=.25e6,
                           get_cmd=None,
                           set_cmd=self._set_span,
                           vals=vals.Numbers())
        self.add_parameter('avg',
                           label='Averages',
                           initial_val=1,
                           get_cmd=None,
                           set_cmd=None,
                           vals=vals.Ints())
        self.add_parameter('power',
                           label='Power ',
                           unit='dBm',
                           initial_value=0,
                           get_cmd=self._get_power_at_freq,
                           set_cmd=False)
        self.add_parameter('ref_lvl',
                           label='Reference power ',
                           unit='dBm',
                           initial_value=0,
                           get_cmd=None,
                           set_cmd=partial(self._set_and_update_config, 'ref_lvl'),
                           vals=vals.Numbers(max_value=20))
        self.add_parameter('external_reference',
                           get_cmd=None,
                           set_cmd=partial(self._set_and_update_config, 'external_reference'),
                           initial_value=False,
                           vals=vals.Bool())
        self.add_parameter('device_type',
                           set_cmd=False,
                           get_cmd=self._get_device_type)
        self.add_parameter('device_mode',
                           initial_value='sweeping',
                           get_cmd=None,
                           set_cmd=partial(self._set_and_update_config, 'device_mode'),
                           vals=vals.Enum('sweeping', 'IQ'))
        self.add_parameter('acquisition_mode',
                           get_cmd=None,
                           set_cmd=False,
                           initial_value='average',
                           vals=vals.Enum('average'))
        self.add_parameter('scale',
                           get_cmd=None,
                           set_cmd=partial(self._set_and_update_config, 'scale'),
                           initial_value='log-scale',
                           vals=vals.Enum('log-scale', 'lin-scale',
                                          'log-full-scale', 'lin-full-scale'))
        self.add_parameter('running',
                           get_cmd=None,
                           set_cmd=None,
                           initial_value=False,
                           vals=vals.Bool())
        self.add_parameter('decimation',
                           get_cmd=None,
                           set_cmd=partial(self._set_and_update_config, 'decimation'),
                           initial_value=1,
                           vals=vals.Ints(1, 8))
        self.add_parameter('bandwidth',
                           label='Bandwidth',
                           unit='Hz',
                           initial_value=0,
                           get_cmd=None,
                           set_cmd=partial(self._set_and_update_config, 'bandwidth'),
                           vals=vals.Numbers())
        # rbw Resolution bandwidth in Hz. RBW can be arbitrary.
        self.add_parameter('rbw',
                           label='Resolution Bandwidth',
                           unit='Hz',
                           initial_value=1e3,
                           get_cmd=None,
                           set_cmd=partial(self._set_and_update_config, 'rbw'),
                           vals=vals.Numbers())
        # vbw Video bandwidth in Hz. VBW must be less than or equal to RBW.
        #  VBW can be arbitrary. For best performance use RBW as the VBW.
        self.add_parameter('vbw',
                           label='Video Bandwidth',
                           unit='Hz',
                           initial_value=1e3,
                           get_cmd=None,
                           set_cmd=partial(self._set_and_update_config, 'vbw'),
                           vals=vals.Numbers())
        self.add_parameter(name='trace',
                           span=self.start(),
                           center=self.frequency(),
                           npts=self.npts(),
                           parameter_class=FrequencySweep)

        self.openDevice()
        self.initialisation()
        self.device_type()

        t1 = time()
        # poor-man's connect_message. We could overwrite get_idn
        # instead and use connect_message.
        print('Initialized SignalHound in %.2fs' % (t1-t0))

    def _update_trace(self)
        sweep_info = self.QuerySweep()
        self.trace.set_sweep(*sweep_info)

    def _set_span(self, val):
        self.configure(span=val)
        self._update_trace()

    def _set_frequency(self, val):
        self.configure(center=val)
        self._update_trace()

    def _set_and_update_config(self, param_name, param_val):
        self.configure(param_name=param_val)

    def configure(self, center=None, span=None, device_mode=None,
                  acquisition_mode=None, scale=None, external_reference=None,
                  ref_lvl=None, vbw=None, rbw=None, decimation=None,
                  bandwidth=None, rejection=True):
        """
        Configure consists of five parts
            1. Center span configuration (freqs and span)
            2. Acquisition configuration
                lin-scale/log-scale
                avg/max power
            3. Configuring the external 10MHz refernce
            4. Configuration of the mode that is being used
            5. Configuration of the tracking generator (not implemented)
                used in VNA mode

        Configure sets the configuration of the instrument using the parameters
        specified in the Qcodes instrument.

        Note that to ensure loading call self.initialisation()
        These two functions are combined in prepare_for_measurement()
        """
        frequency = center or self.frequency()
        span = span or self.span()
        device_mode = device_mode or self.device_mode()
        acquisition_mode = acquisition_mode or self.acquisition_mode()
        ref_lvl = ref_lvl or self.ref_lvl()
        rbw = rbw or self.rbw()
        vbw = vbw or self.vbw()
        scale = scale or self.scale()
        decimation = decimation or self.decimation()
        bandwidth = bandwidth or self.bandwidth()
        external_reference = external_reference or self.external_reference()

        # 1. CenterSpan Configuration
        center = ct.c_double(frequency)
        span = ct.c_double(span)
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
        detector = detectorVals[acquisition_mode]
        scale = scaleVals[scale]

        err = self.dll.saConfigAcquisition(self.deviceHandle, detector, scale)
        self.check_for_error(err)

        # 3. Reference Level configuration
        log.info('Setting device reference level configuration.')
        err = self.dll.saConfigLevel(
            self.deviceHandle, ct.c_double(ref_lvl))
        self.check_for_error(err)

        # 4. External Reference configuration
        if external_reference:
            log.info('Setting reference frequency from external source.')
            err = self.dll.saEnableExternalReference(self.deviceHandle)
            self.check_for_error(err)

        if device_mode == 'sweeping':    
            reject_var = ct.c_bool(rejection)
            log.info('Setting device Sweeping configuration.')
            err = self.dll.saConfigSweepCoupling(
                self.deviceHandle, ct.c_double(rbw),
                ct.c_double(vbw), reject_var)
            self.check_for_error(err)
        elif device_mode == 'IQ':
            raise NotImplementedError
            err = self.dll.saConfigIQ(
                self.deviceHandle, ct.c_int(decimation),
                ct.c_double(bandwidth))
            self.check_for_error(err)
        return

    @classmethod
    def default_server_name(cls, **kwargs):
        return 'USB'

    def openDevice(self):
        log.info('Opening Device')
        self.deviceHandle = ct.c_int(0)
        deviceHandlePnt = ct.pointer(self.deviceHandle)
        ret = self.dll.saOpenDevice(deviceHandlePnt)
        if ret != self.saStatus['saNoError']:
            if ret == self.saStatus['saNullPtrErr']:
                raise ValueError('Could not open device due to '
                                 'null-pointer error!')
            elif ret == self.saStatus['saDeviceNotOpenErr']:
                raise ValueError('Could not open device!')
            else:
                raise ValueError('Could not open device due to unknown '
                                 'reason! Error = %d' % ret)

        self.devOpen = True
        self.device_type()

    def closeDevice(self):
        log.info('Closing Device with handle num: '
                 f'{self.deviceHandle.value}')

        try:
            self.dll.saAbort(self.deviceHandle)
            log.info('Running acquistion aborted.')
        except Exception as e:
            log.info(f'Could not abort acquisition: {e}')

        ret = self.dll.saCloseDevice(self.deviceHandle)
        if ret != self.saStatus['saNoError']:
            raise ValueError('Error closing device!')
        log.info(f'Closed Device with handle num: {self.deviceHandle.value}')
        self.devOpen = False
        self.running(False)

    def abort(self):
        log.info('Stopping acquisition')

        err = self.dll.saAbort(self.deviceHandle)
        if err == self.saStatus['saNoError']:
            log.info('Call to abort succeeded.')
            self.running(False)
        elif err == self.saStatus['saDeviceNotOpenErr']:
            raise IOError('Device not open!')
        elif err == self.saStatus['saDeviceNotConfiguredErr']:
            raise IOError('Device was already idle! Did you call abort '
                          'without ever calling initiate()?')
        else:
            raise IOError('Unknown error setting abort! Error = %s' % err)

    def preset(self):
        log.warning('Performing hardware-reset of device!')
        log.warning('Please ensure you close the device handle within '
                    'two seconds of this call!')

        err = self.dll.saPreset(self.deviceHandle)
        if err == self.saStatus['saNoError']:
            log.info('Call to preset succeeded.')
        elif err == self.saStatus['saDeviceNotOpenErr']:
            raise IOError('Device not open!')
        else:
            raise IOError(f'Unknown error calling preset! Error = {err}')

    def _get_device_type(self):
        log.info('Querying device for model information')

        devType = ct.c_uint(0)
        devTypePnt = ct.pointer(devType)

        err = self.dll.saGetDeviceType(self.deviceHandle, devTypePnt)
        if err == self.saStatus['saNoError']:
            pass
        elif err == self.saStatus['saDeviceNotOpenErr']:
            raise IOError('Device not open!')
        elif err == self.saStatus['saNullPtrErr']:
            raise IOError('Null pointer error!')
        else:
            raise IOError('Unknown error setting getDeviceType! '
                          'Error = %s' % err)

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

    def initialisation(self, flag=0):
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
        if err == self.saStatus['saNoError']:
            self.running(True)
            log.info('Call to initiate succeeded.')
        elif err == self.saStatus['saDeviceNotOpenErr']:
            raise IOError('Device not open!')
        elif err == self.saStatus['saInvalidParameterErr']:
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
        elif err == self.saStatus['saBandwidthErr']:
            raise IOError('RBW is larger than your span. (Sweep Mode)!')
        self.check_for_error(err)
        return

    def QuerySweep(self):
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
        if err == self.saStatus['saNoError']:
            pass
        elif err == self.saStatus['saDeviceNotOpenErr']:
            raise IOError('Device not open!')
        elif err == self.saStatus['saDeviceNotConfiguredErr']:
            raise IOError('The device specified is not currently streaming!')
        elif err == self.saStatus['saNullPtrErr']:
            raise IOError('Null pointer error!')
        else:
            raise IOError('Unknown error!')

        info = [sweep_len.value, start_freq.value, stepsize.value]
        return info

    def _get_sweep_data(self):
        """
        This function performs a sweep over the configured ranges.
        The result of the sweep is returned along with the sweep points

        returns:
            datamin numpy array
        """
        try:
            sweep_len, start_freq, stepsize = self.QuerySweep()
        except:
            self.prepare_for_measurement()
            sweep_len, start_freq, stepsize = self.QuerySweep()

        end_freq = start_freq + stepsize*(sweep_len-1)

        minarr = (ct.c_float * sweep_len)()
        maxarr = (ct.c_float * sweep_len)()
        sleep(.1)  # Added extra sleep for updating issue
        err = self.dll.saGetSweep_32f(self.deviceHandle, minarr, maxarr)
        sleep(.1)  # Added extra sleep
        if not err == self.saStatus['saNoError']:
            # if an error occurs tries preparing the device and then asks again
            print('Error raised in QuerySweepInfo, preparing for measurement')
            sleep(.1)
            self.prepare_for_measurement()
            sleep(.1)
            minarr = (ct.c_float * sweep_len)()
            maxarr = (ct.c_float * sweep_len)()
            err = self.dll.saGetSweep_32f(self.deviceHandle, minarr, maxarr)

        if err == self.saStatus['saNoError']:
            pass
        elif err == self.saStatus['saDeviceNotOpenErr']:
            raise IOError('Device not open!')
        elif err == self.saStatus['saDeviceNotConfiguredErr']:
            raise IOError('The device specified is not currently streaming!')
        elif err == self.saStatus['saNullPtrErr']:
            raise IOError('Null pointer error!')
        elif err == self.saStatus['saInvalidModeErr']:
            raise IOError('Invalid mode error!')
        elif err == self.saStatus['saCompressionWarning']:
            raise IOError('Input voltage overload!')
        elif err == self.saStatus['sCUSBCommErr']:
            raise IOError('Error ocurred in the USB connection!')
        else:
            raise IOError('Unknown error!')

        datamin = np.array([minarr[elem] for elem in range(sweep_len)])

        return datamin

    def _get_averaged_sweep_data(self):
        """
        Averages over SH.sweep Navg times

        """
        data = np.zeros(self.npts())
        Navg = self.avg()
        for i in range(Navg):
            data += self._get_sweep_data()
        return data / Navg

    def _get_power_at_freq(self, Navg=1):
        """
        Returns the maximum power in a window of 250kHz
        around the specified  frequency.
        The integration window is specified by the VideoBandWidth (set by vbw)
        
        NB THIS IS A LIE, would be great if this function existed but what this actually
            did was to take a sweep and take
            the maximum power in the sweep NOT in a 250kHz window
        """
        original_mode = self.device_mode()
        self.device_mode('IQ')
        poweratfreq = 0
        data = self._get_averaged_sweep_data()
        max_power = np.max(data)
        self.device_mode(original_mode)
        return max_power

    def prepare_for_measurement(self):
        # TODO: is this really necessary?
        self.configure()
        self.initialisation()
        return

    def safe_reload(self):
        self.closeDevice()
        self.reload()

    def check_for_error(self, err):
        if err != self.saStatus['saNoError']:
            err_msg = self.saStatus_inverted[err]
            if err > 0:
                print('Warning:', err_msg)
            else:
                raise IOError(err_msg)


class constants:
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
