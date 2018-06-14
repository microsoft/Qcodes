from time import sleep, time
import numpy as np
import ctypes as ct
import logging

from qcodes import Instrument, validators as vals


class SignalHound_USB_SA124B(Instrument):
    '''
    This is a direct port of the signal hound QTLab driver by Ramiro
    Edited by Adriaan Rol

    Status: Beta version.
        This driver is functional but not all features have been implemented.

    TODO:
        Add tracking generator mode
    '''
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

        self.log = logging.getLogger('Main.DeviceInt')
        logging.info(__name__ +
                     ' : Initializing instrument SignalHound USB 124A')
        self.dll = ct.CDLL(dll_path or self.dll_path)
        self.hf = constants

        self.add_parameter('frequency',
                           label='Frequency ',
                           unit='Hz',
                           initial_value=5e9,
                           get_cmd=None, set_cmd=None,
                           vals=vals.Numbers())
        self.add_parameter('span',
                           label='Span ',
                           unit='Hz',
                           initial_value=.25e6,
                           get_cmd=None, set_cmd=None,
                           vals=vals.Numbers())
        self.add_parameter('power',
                           label='Power ',
                           unit='dBm',
                           initial_value=0,
                           get_cmd=None, set_cmd=None,
                           vals=vals.Numbers(max_value=20))
        self.add_parameter('ref_lvl',
                           label='Reference power ',
                           unit='dBm',
                           initial_value=0,
                           get_cmd=None, set_cmd=None,
                           vals=vals.Numbers(max_value=20))
        self.add_parameter('external_reference',
                           get_cmd=None, set_cmd=None,
                           initial_value=False,
                           vals=vals.Bool())
        self.add_parameter('device_type',
                           get_cmd=self._do_get_device_type)

        self.add_parameter('device_mode',
                           initial_value='sweeping',
                           get_cmd=None, set_cmd=None,
                           vals=vals.Anything())
        self.add_parameter('acquisition_mode',
                           get_cmd=None, set_cmd=None,
                           initial_value='average',
                           vals=vals.Enum('average', 'min-max'))
        self.add_parameter('scale',
                           get_cmd=None, set_cmd=None,
                           initial_value='log-scale',
                           vals=vals.Enum('log-scale', 'lin-scale',
                                          'log-full-scale', 'lin-full-scale'))
        self.add_parameter('running',
                           get_cmd=None, set_cmd=None,
                           initial_value=False,
                           vals=vals.Bool())
        self.add_parameter('decimation',
                           get_cmd=None, set_cmd=None,
                           initial_value=1,
                           vals=vals.Ints(1, 8))
        self.add_parameter('bandwidth',
                           label='Bandwidth',
                           unit='Hz',
                           initial_value=0,
                           get_cmd=None, set_cmd=None,
                           vals=vals.Numbers())
        # rbw Resolution bandwidth in Hz. RBW can be arbitrary.
        self.add_parameter('rbw',
                           label='Resolution Bandwidth',
                           unit='Hz',
                           initial_value=1e3,
                           get_cmd=None, set_cmd=None,
                           vals=vals.Numbers())
        # vbw Video bandwidth in Hz. VBW must be less than or equal to RBW.
        #  VBW can be arbitrary. For best performance use RBW as the VBW.
        self.add_parameter('vbw',
                           label='Video Bandwidth',
                           unit='Hz',
                           initial_value=1e3,
                           get_cmd=None, set_cmd=None,
                           vals=vals.Numbers())

        self.openDevice()
        self.device_type()

        t1 = time()
        print('Initialized SignalHound in %.2fs' % (t1-t0))

    @classmethod
    def default_server_name(cls, **kwargs):
        return 'USB'

    def openDevice(self):
        self.log.info('Opening Device')
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
        self.get('device_type')

    def closeDevice(self):
        self.log.info('Closing Device with handle num: ',
                      self.deviceHandle.value)

        try:
            self.dll.saAbort(self.deviceHandle)
            self.log.info('Running acquistion aborted.')
        except Exception as e:
            self.log.info('Could not abort acquisition: %s', e)

        ret = self.dll.saCloseDevice(self.deviceHandle)
        if ret != self.saStatus['saNoError']:
            raise ValueError('Error closing device!')
        print('Closed Device with handle num: ', self.deviceHandle.value)
        self.devOpen = False
        self.running(False)

    def abort(self):
        self.log.info('Stopping acquisition')

        err = self.dll.saAbort(self.deviceHandle)
        if err == self.saStatus['saNoError']:
            self.log.info('Call to abort succeeded.')
            self.running(False)
        elif err == self.saStatus['saDeviceNotOpenErr']:
            raise IOError('Device not open!')
        elif err == self.saStatus['saDeviceNotConfiguredErr']:
            raise IOError('Device was already idle! Did you call abort '
                          'without ever calling initiate()?')
        else:
            raise IOError('Unknown error setting abort! Error = %s' % err)

    def preset(self):
        self.log.warning('Performing hardware-reset of device!')
        self.log.warning('Please ensure you close the device handle within '
                         'two seconds of this call!')

        err = self.dll.saPreset(self.deviceHandle)
        if err == self.saStatus['saNoError']:
            self.log.info('Call to preset succeeded.')
        elif err == self.saStatus['saDeviceNotOpenErr']:
            raise IOError('Device not open!')
        else:
            raise IOError('Unknown error calling preset! Error = %s' % err)

    def _do_get_device_type(self):
        self.log.info('Querying device for model information')

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
        mode = self.get('device_mode')
        modeOpts = {
            'sweeping': self.hf.sa_SWEEPING,
            'real_time': self.hf.sa_REAL_TIME,
            'IQ': self.hf.sa_IQ,  # not implemented
            'idle': self.hf.sa_IDLE  # not implemented
        }
        if mode in modeOpts:
            mode = modeOpts[mode]
        else:
            raise ValueError('Mode must be one of %s. Passed value was %s.' %
                             (modeOpts, mode))
        err = self.dll.saInitiate(self.deviceHandle, mode, flag)

        ###################################
        # Below here only error handling
        ###################################
        if err == self.saStatus['saNoError']:
            self.running(True)
            self.log.info('Call to initiate succeeded.')
        elif err == self.saStatus['saDeviceNotOpenErr']:
            raise IOError('Device not open!')
        elif err == self.saStatus['saInvalidParameterErr']:
            print('saInvalidParameterErr!')
            print('In real-time mode, this value may be returned if the span',
                  'limits defined in the API header are broken. Also in',
                  'real-time mode, this error will be returned if the',
                  ' resolution bandwidth is outside the limits defined in',
                  ' the API header.')
            print('In time-gate analysis mode this error will be returned if',
                  ' span limits defined in the API header are broken. Also in',
                  ' time gate analysis, this error is returned if the',
                  ' bandwidth provided require more samples for processing',
                  ' than is allowed in the gate length. To fix this, ',
                  'increase rbw/vbw.')
            raise IOError('The value for mode did not match any known value.')
        # This error code does not exist!??
        # elif err == self.saStatus['saAllocationLimitError']:
        #     print('This value is returned in extreme circumstances. The API',
        #           ' currently limits the amount of RAM usage to 1GB. When',
        #           ' exceptional parameters are provided, such as very low ',
        #           'bandwidths, or long sweep times, this error may be ',
        #           'returned. At this point you have reached the boundaries of',
        #           ' the device. The processing algorithms are optimized for',
        #           ' speed at the expense of space, which is the reason',
        #           ' this can occur.''')
        #     raise IOError('Could not allocate sufficent RAM!')
        elif err == self.saStatus['saBandwidthErr']:
            raise IOError('RBW is larger than your span. (Sweep Mode)!')
        self.check_for_error(err)
        # else:
        #     raise IOError('Unknown error setting initiate! Error = %s' % err)

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

    def configure(self, rejection=True):
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
        # CenterSpan Configuration
        frequency = self.get('frequency')
        span = self.get('span')
        center = ct.c_double(frequency)
        span = ct.c_double(span)
        self.log.info('Setting device CenterSpan configuration.')

        err = self.dll.saConfigCenterSpan(self.deviceHandle, center, span)
        self.check_for_error(err)

        # Acquisition configuration
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
        if self.acquisition_mode() in detectorVals:
            detector = detectorVals[self.acquisition_mode()]
        else:
            raise ValueError('Invalid Detector mode! Detector  must be one of '
                             '%s. Specified detector = %s' %
                             (list(detectorVals.keys()), detector))
        if self.scale() in scaleVals:
            scale = scaleVals[self.scale()]
        else:
            raise ValueError('Invalid Scaling mode! Scaling mode must be one '
                             'of %s. Specified scale = %s' %
                             (list(scaleVals.keys()), scale))
        err = self.dll.saConfigAcquisition(self.deviceHandle, detector, scale)
        self.check_for_error(err)

        # Reference Level configuration
        self.log.info('Setting device reference level configuration.')
        err = self.dll.saConfigLevel(
            self.deviceHandle, ct.c_double(self.get('ref_lvl')))
        self.check_for_error(err)

        # External Reference configuration
        if self.external_reference():
            self.log.info('Setting reference frequency from external source.')
            err = self.dll.saEnableExternalReference(self.deviceHandle)
            self.check_for_error(err)

        if self.device_mode() == 'sweeping':
            # Sweeping Configuration
            reject_var = ct.c_bool(rejection)
            self.log.info('Setting device Sweeping configuration.')
            err = self.dll.saConfigSweepCoupling(
                self.deviceHandle, ct.c_double(self.get('rbw')),
                ct.c_double(self.get('vbw')), reject_var)
            self.check_for_error(err)
        elif self.device_mode() == 'IQ':
            err = self.dll.saConfigIQ(
                self.deviceHandle, ct.c_int(self.get('decimation')),
                ct.c_double(self.get('bandwidth')))
            self.check_for_error(err)
        return

    def sweep(self):
        """
        This function performs a sweep over the configured ranges.
        The result of the sweep is returned along with the sweep points

        returns:

        """
        try:
            sweep_len, start_freq, stepsize = self.QuerySweep()
        except:
            self.prepare_for_measurement()
            sweep_len, start_freq, stepsize = self.QuerySweep()

        end_freq = start_freq + stepsize*(sweep_len-1)
        freq_points = np.linspace(start_freq, end_freq, sweep_len)

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

        # note if used in averaged mode (set in config) datamin=datamax
        datamin = np.array([minarr[elem] for elem in range(sweep_len)])
        datamax = np.array([minarr[elem] for elem in range(sweep_len)])

        return np.array([freq_points, datamin, datamax])

    def get_power_at_freq(self, Navg=1):
        '''
        Returns the maximum power in a window of 250kHz
        around the specified  frequency.
        The integration window is specified by the VideoBandWidth (set by vbw)
        '''
        poweratfreq = 0
        for i in range(Navg):
            data = self.sweep()
            max_power = np.max(data[1][:])
            poweratfreq += max_power
        self.power(poweratfreq / Navg)
        return self.power()

    def get_spectrum(self, Navg=1):
        """
        Averages over SH.sweep Navg times

        """
        sweep_params = self.QuerySweep()
        data_spec = np.zeros(sweep_params[0])
        for i in range(Navg):
            data = self.sweep()
            data_spec[:] += data[1][:]
        data_spec[:] = data_spec[:] / Navg
        sweep_points = data[0][:]
        return np.array([sweep_points, data_spec])

    def prepare_for_measurement(self):
        self.set('device_mode', 'sweeping')
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
