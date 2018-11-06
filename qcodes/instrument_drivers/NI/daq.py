import ctypes
import logging
#import types
import os
import sys
from numpy import zeros, mean, float64, float32

from qcodes.instrument.base import Instrument
from qcodes.instrument.base import Parameter

if os.name == 'nt':
    nidaq = ctypes.windll.nicaiu
    NIDAQ_dll = ctypes.windll.nicaiu
elif os.name == 'posix':
    # not supported
    nidaq = ctypes.cdll.LoadLibrary("libnidaqmx.so")
else:
    print('Operating system not supported.')

int32 = ctypes.c_long
uInt32 = ctypes.c_ulong
uInt64 = ctypes.c_ulonglong
cfloat64 = ctypes.c_double
TaskHandle = uInt32

DAQmx_Val_Cfg_Default = int32(-1)

DAQmx_Val_RSE               = 10083
DAQmx_Val_NRSE              = 10078
DAQmx_Val_Diff              = 10106
DAQmx_Val_PseudoDiff        = 12529

_config_map = {
    'DEFAULT': DAQmx_Val_Cfg_Default,
    'RSE': DAQmx_Val_RSE,
    'NRSE': DAQmx_Val_NRSE,
    'DIFF': DAQmx_Val_Diff,
    'PSEUDODIFF': DAQmx_Val_PseudoDiff,
}

DAQmx_Val_Volts             = 10348
DAQmx_Val_Rising            = 10280
DAQmx_Val_Falling           = 10171
DAQmx_Val_FiniteSamps       = 10178
DAQmx_Val_GroupByChannel    = 0
DAQmx_Val_GroupByScanNumber = 1

DAQmx_Val_CountUp           = 10128
DAQmx_Val_CountDown         = 10124
DAQmx_Val_ExtControlled     = 10326

NIDAQ_dll.DAQmxGetErrorString.restype         = int32
NIDAQ_dll.DAQmxGetDevProductType.restype      = int32

NIDAQ_dll.DAQmxCreateAIVoltageChan.restype    = int32 
NIDAQ_dll.DAQmxCreateAIVoltageChan.argtype = [
            TaskHandle,
            ctypes.c_char_p, #const char physicalChannel[],
            ctypes.c_char_p, #const char nameToAssignToChannel[],
            int32, # terminalConfig
            cfloat64, # minVal
            cfloat64, # maxVal
            int32, # units
            ctypes.c_char_p, #const char customScaleName[]
            ]

def CHK(err): 
    """Error checking routine"""
    if err < 0:
        buf_size = 100
        buf = ctypes.create_string_buffer(buf_size)
        NIDAQ_dll.DAQmxGetErrorString(err, 
            ctypes.byref(buf), buf_size)
        raise RuntimeError('Nidaq call failed with error %d: %s' % \
            (err, repr(buf.value)))

def buf_to_list(buf):
    """Convert a string buffer to a list of contents"""
    name = ''
    namelist = []
    buf_stripped = buf.raw.decode().rstrip('\x00')
#    for ch in buf_stripped:
#        if (ch == '0') or (ch == '\t') or (ch == '\n'):
#            name = name.rstrip(',')
#            if len(name) > 0:
#                namelist.append(name)
#                name = ''
#            if ch == '\000':
#                break
#        else:
#            name += ch
#
#    return namelist
    return buf_stripped.split(', ')

class NI_DAQ(Instrument):
    """
    This is the driver for the National Instruments data acquisition
    instruments
    """

    @classmethod
    def find_daqs(cls, dll_path=None):
        """
        Find connected National Instruments DAQs
        """
        # does a function like this exist in DAQmx?
        dll = ctypes.windll.nicaiu
        bufsize = 1024
        buf = ctypes.create_string_buffer(bufsize)
        dll.DAQmxGetSysDevNames(ctypes.byref(buf), bufsize)
        return buf_to_list(buf)

    def __init__(self, name, dev_id, **kwargs):
        super().__init__(name, **kwargs)

        self.dev_id = dev_id
        if not dev_id in self.find_daqs():
            raise ValueError('DAQ with id {0} not present in system.'.format(dev_id))
        self.ai_channels = self.get_physical_input_channels()
        self.ao_channels = self.get_physical_output_channels()

    def get_device_type(self):
        """Get the type of a DAQ device."""
        # int32 __CFUNC DAQmxGetDevProductType(const char device[], char *data,
        # uInt32 bufferSize)
        bufsize = 1024
        buf = ctypes.create_string_buffer(bufsize)
        NIDAQ_dll.DAQmxGetDevProductType(
            self.dev_id.encode('ascii'), ctypes.byref(buf), bufsize) 
        return buf_to_list(buf)[0]

    def get_physical_input_channels(self):
        """Return a list of physical input channels on a device."""
        bufsize = 1024
        buf = ctypes.create_string_buffer(bufsize)
        CHK(NIDAQ_dll.DAQmxGetDevAIPhysicalChans(
            self.dev_id.encode('ascii'), 
            ctypes.byref(buf), bufsize))
        channel_list = buf_to_list(buf)
        channel_list = [channel.lstrip(self.dev_id+'/') for channel in channel_list]
        return channel_list

    def get_physical_output_channels(self):
        """Return a list of physical output channels on a device."""
        bufsize = 1024
        buf = ctypes.create_string_buffer(bufsize)
        NIDAQ_dll.DAQmxGetDevAOPhysicalChans(self.dev_id.encode('ascii'), 
            ctypes.byref(buf), bufsize)
        channel_list = buf_to_list(buf)
        channel_list = [channel.lstrip(self.dev_id+'/') for channel in channel_list]
        return channel_list

    def get_input_voltage_ranges(self):
        """Get the available voltage ranges for the analog inputs"""
        bufsize = 32
        range_list_type = cfloat64 * bufsize
        range_list = range_list_type()
        NIDAQ_dll.DAQmxGetDevAIVoltageRngs(self.dev_id.encode('ascii'), 
            ctypes.byref(range_list), uInt32(bufsize))
        range_list = list(range_list)
        range_values_n = range_list.index(0.0)
        range_n = range_values_n / 2
        return_list = []
        for idx in range(range_n):
            return_list.append([range_list[2*idx],
                                range_list[(2*idx)+1]])        
        return return_list

    def get_maximum_input_channel_rate(self):
        """Get the highest sample rate for a single input channel"""
        sample_rate = cfloat64()
        NIDAQ_dll.DAQmxGetDevAIMaxSingleChanRate(self.dev_id.encode('ascii'), 
                                                 ctypes.byref(sample_rate))
        return sample_rate.value

    def read_analog(self, devchan, samples, sample_rate, timeout, chan_config, 
                    minv=-10.0, maxv=10.0, triggered=False, averaging=True):
        """
        Input:
            samples
            
        Output:
            A numpy.array with the data on success, None on error

        """
        devchan = (self.dev_id + '/' + devchan).encode('ascii')
        timeout = (timeout + samples/sample_rate)

        if type(chan_config) is str:
            if chan_config in _config_map:
                config = _config_map[chan_config]
            else:
                raise ValueError('Config {0} not available.'.format(
                    chan_config))
        else:
            config = chan_config

        data = zeros(samples, dtype=float64)

        with NIDAQmx_task() as taskHandle:
            read = int32()

            CHK(NIDAQ_dll.DAQmxCreateTask("", ctypes.byref(taskHandle)))
            CHK(NIDAQ_dll.DAQmxCreateAIVoltageChan(
                taskHandle, 
                devchan, 
                b"",
                config,
                cfloat64(minv), 
                cfloat64(maxv),
                int32(DAQmx_Val_Volts), 
                b""))

            if samples > 1:
                CHK(NIDAQ_dll.DAQmxCfgSampClkTiming(
                    taskHandle, "", 
                    cfloat64(sample_rate),
                    int32(DAQmx_Val_Rising), 
                    int32(DAQmx_Val_FiniteSamps),
                    uInt64(samples)));
                
                if triggered:
                    if trigger_slope == 'POS':
                        slope = DAQmx_Val_Rising
                    elif trigger_slope == 'NEG':
                        slope = DAQmx_Val_Falling
                    else:
                        raise ValueError('Use POS or NEG for the trigger slope')
                    CHK(NIDAQ_dll.DAQmxCfgDigEdgeRefTrig(
                                    taskHandle,
                                    "/Dev1/PFI0",
                                    slope,
                                    uInt32(pre_trig_samples)))
                    
                CHK(NIDAQ_dll.DAQmxStartTask(taskHandle))
                CHK(NIDAQ_dll.DAQmxReadAnalogF64(
                    taskHandle, 
                    samples, 
                    cfloat64(timeout),
                    DAQmx_Val_GroupByChannel, 
                    data.ctypes.data,
                    samples, 
                    ctypes.byref(read), None))
            else:
                CHK(NIDAQ_dll.DAQmxReadAnalogScalarF64(
                    taskHandle, cfloat64(timeout),
                    data.ctypes.data, None))
                read = int32(1)

        if read.value > 0:
            if samples == 1:
                return data[0]
            else:
                if averaging:
                    return mean(data)
                else:
                    return data
        else:
            logging.warning('No values were read.')



class DAQReadAnalog(Parameter):
    """
    This class represents the analog acquisition

        Input:
            devchan (string): device/channel specifier, such as Dev1/ai0
            minv (float): the minimum voltage
            maxv (float): the maximum voltage
            config (string or int): the configuration of the channel
            triggered (boolean): whether the measurement is triggered
            trigger_slope (string): whether we are using the positive or negative
                                    slope for the trigger.
    """
    def __init__(self, samples=None, name=None, label=None, unit=None, instrument=None,
                 value=None, byte_to_value_dict=None, vals=None, get_cmd=None):
        
        super().__init__(name=name, label=label, unit=unit, vals=vals,
                         instrument=instrument, get_cmd=get_cmd)
        self.instrument=instrument
        self.devchan=name
        self.minv=-10.0
        self.maxv=10.0
        self.timeout=10.0
        self.chan_config=DAQmx_Val_Cfg_Default
        self.averaging=True
        self.triggered = False
        self.trigger_slope='POS'
        self.pre_trig_samples=0

    def get(self):
        """Get an analog input value.

        Must return a single value, so either averaging is True or samples=1

        If multiple values are required, use get_array
        """
        samples =     getattr(self.instrument, self.devchan+'_samples')()
        if (not self.averaging) and (samples>1):
            raise ValueError('For multiple samples averaging must be on.')
        sample_rate = getattr(self.instrument, self.devchan+'_sample_rate')()
        timeout =     getattr(self.instrument, self.devchan+'_timeout')()
        return self.instrument.read_analog(self.devchan,
                                           samples,
                                           sample_rate,
                                           timeout,
                                           self.chan_config,
                                           self.minv,
                                           self.maxv,
                                           self.triggered,
                                           self.averaging,
                                            )

    def get_array(self):
        """Get an array of acquired values."""
        samples =     getattr(self.instrument, self.devchan+'_samples')()
        sample_rate = getattr(self.instrument, self.devchan+'_sample_rate')()
        timeout =     getattr(self.instrument, self.devchan+'_timeout')()
        return self.instrument.read_analog(self.devchan,
                                           samples,
                                           sample_rate,
                                           timeout,
                                           self.chan_config,
                                           self.minv,
                                           self.maxv,
                                           self.triggered,
                                           False
                                            )

class DAQWriteAnalog(Parameter):
    """Write a value to an analog output.

    Args:
        name
        label
        unit
        instrument
        value
        byte_to_value_dict
        vals
        get_cmd
    """
    def __init__(self, name=None, label=None, unit=None, instrument=None,
                 value=None, byte_to_value_dict=None, vals=None, get_cmd=None):
        
        super().__init__(name=name, label=label, unit=unit, vals=vals,
                         instrument=instrument, get_cmd=get_cmd)
        self.sample_rate=10000.0
        self.minv=-10.0
        self.maxv=10.0
        self.timeout=10.0

class NIDAQmx_task:
    """Class to handle the taskhandle."""
    def __init__(self):
        self.taskHandle = TaskHandle(0)

    def __enter__(self):
        return self.taskHandle

    def __exit__(self, type, value, traceback):
        if self.taskHandle.value != 0:
            NIDAQ_dll.DAQmxStopTask(self.taskHandle)
            NIDAQ_dll.DAQmxClearTask(self.taskHandle)
