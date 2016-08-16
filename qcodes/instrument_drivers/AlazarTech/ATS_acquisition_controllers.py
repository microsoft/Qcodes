from .ATS import AcquisitionController
import math
import numpy as np
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals


class HD_Controller(AcquisitionController):
    """Heterodyne Measurement Controller
    Does averaged DFT on 2 channel Alazar measurement 

    TODO(nataliejpg) handling of channel number
    TODO(nataliejpg) test angle data
    """
    def __init__(self, freq_dif):
        self.freq_dif = freq_dif
        self.samples_per_record = None
        self.records_per_buffer = None
        self.buffers_per_acquisition = None
        self.number_of_channels = 2
        self.buffer = None

    def pre_start_capture(self, alazar):
        """Get config data from alazar card and set up DFT"""
        self.samples_per_record = alazar.samples_per_record()
        self.records_per_buffer = alazar.records_per_buffer()
        self.buffers_per_acquisition = alazar.buffers_per_acquisition()
        self.sample_rate = alazar.get_sample_speed()
        self.buffer = np.zeros(self.samples_per_record *
                              self.records_per_buffer *
                              self.number_of_channels)
        
        # TODO(nataliejpg) leave super explicit or save lines? add error/logging?
        averaging = self.buffers_per_acquisition * self.records_per_buffer
        record_duration = self.samples_per_record/self.sample_rate
        time_period_dif = 1/self.freq_dif
        cycles_measured = record_duration / time_period_dif
        oversampling_rate = self.sample_rate/(2*self.freq_dif)
        print("Average over {} records".format(averaging))
        print("Oscillations per record: {} (expect 100+)".format(cycles_measured))
        print("Oversampling rate: {} (expect > 2)".format(oversampling_rate))

        integer_list = np.arange(self.samples_per_record)
        angle_list = (2 * np.pi * self.freq_dif / self.sample_rate *
                      integer_list)
        self.cos_list = np.cos(angle_list)
        self.sin_list = np.sin(angle_list)
                              
    def pre_acquire(self, alazar):
        # gets called after 'AlazarStartCapture'
        pass

    def handle_buffer(self, alazar, data):
        self.buffer += data

    def post_acquire(self, alazar):
        """Average over records in buffer and do DFT:
        assumes samples are arranged in the buffer as follows:
        S0A, S0B, ..., S1A, S1B, ...
        with SXY the sample number X of channel Y.
        """
        records_per_acquisition = (1. * self.buffers_per_acquisition *
                                   self.records_per_buffer)
        recordA = np.zeros(self.samples_per_record)
        for i in range(self.records_per_buffer):
            i0 = i * self.samples_per_record * self.number_of_channels
            i1 = i0 + self.samples_per_record * self.number_of_channels
            recordA += self.buffer[i0:i1:self.number_of_channels] / records_per_acquisition
        recordB = np.zeros(self.samples_per_record)
        for i in range(self.records_per_buffer):
            i0 = (i * self.samples_per_record * self.number_of_channels) + 1
            i1 = i0 + self.samples_per_record *self.number_of_channels
            recordB += self.buffer[i0:i1:self.number_of_channels] / records_per_acquisition
        
        resA = self.fit(recordA)
        resB = self.fit(recordB)
        
        return resA, resB
      
    def fit(self, rec):
        """Do Discrete Fourier Transform and return magnitude and phase data"""
        RePart = np.dot(rec, self.cos_list) / self.samples_per_record
        ImPart = np.dot(rec, self.sin_list) / self.samples_per_record
        # factor of 2 as amplitude is split between finite term and double frequency term
        ampl = 2 * np.sqrt(RePart ** 2 + ImPart ** 2)
        phase = math.atan2(ImPart, RePart) * 180 / (2 * math.pi)
        
        return [ampl, phase]


class Basic_AcquisitionController(AcquisitionController):
    """Basic AcquisitionController tested on ATS9360
    returns unprocessed data averaged by record with 2 channels
    """
    def __init__(self, name, alazar_id, **kwargs):
        self.samples_per_record = None
        self.records_per_buffer = None
        self.buffers_per_acquisition = None
        self.number_of_channels = 2
        self.buffer = None
        super().__init__(name, alazar_id, **kwargs)

    def pre_start_capture(self, alazar):
        self.samples_per_record = alazar.samples_per_record()
        self.records_per_buffer = alazar.records_per_buffer()
        self.buffers_per_acquisition = alazar.buffers_per_acquisition()
        self.buffer = np.zeros(self.samples_per_record *
                              self.records_per_buffer *
                              self.number_of_channels)

    def pre_acquire(self, alazar):
        # gets called after 'AlazarStartCapture'
        pass

    def handle_buffer(self, alazar, data):
        self.buffer += data

    def post_acquire(self, alazar):
        # average over records in buffer:
        # for ATS9360 samples are arranged in the buffer as follows:
        # S0A, S0B, ..., S1A, S1B, ...
        # with SXY the sample number X of channel Y.
        records_per_acquisition = (1. * self.buffers_per_acquisition *
                                   self.records_per_buffer)
        recordA = np.zeros(self.samples_per_record)
        for i in range(self.records_per_buffer):
            i0 = i * self.samples_per_record * self.number_of_channels
            i1 = i0 + self.samples_per_record * self.number_of_channels
            recordA += self.buffer[i0:i1:self.number_of_channels] / records_per_acquisition

        recordB = np.zeros(self.samples_per_record)
        for i in range(self.records_per_buffer):
            i0 = (i * self.samples_per_record * self.number_of_channels) + 1
            i1 = i0 + self.samples_per_record *self.number_of_channels
            recordB += self.buffer[i0:i1:self.number_of_channels] / records_per_acquisition
        return recordA, recordB


class Average_AcquisitionController(AcquisitionController):
    """Basic AcquisitionController tested on ATS9360
    returns unprocessed data averaged by record with 2 channels
    """
    def __init__(self, name, alazar_id, **kwargs):
        self.samples_per_record = None
        self.records_per_buffer = None
        self.buffers_per_acquisition = None
        self.buffer = None
        self.acquisitionkwargs = {'acquisition_controller': self}
        super().__init__(name, alazar_id, **kwargs)
        self.alazar = self.get_alazar()
        self.add_parameter(name='average_mode',
                           parameter_class=ManualParameter,
                           initial_value='trace',
                           vals=vals.Enum('none', 'trace', 'point'))

    def set_acquisitionkwargs(self, **kwargs):
        self.acquisitionkwargs.update(**kwargs)

    def pre_start_capture(self):
        self.samples_per_record = self.alazar.samples_per_record()
        self.records_per_buffer = self.alazar.records_per_buffer()
        self.buffers_per_acquisition = self.alazar.buffers_per_acquisition()
        self.number_of_channels = len(self.alazar.channel_selection())
        self.buffer = np.zeros(self.samples_per_record *
                              self.records_per_buffer *
                              self.number_of_channels)

    def pre_acquire(self):
        # gets called after 'AlazarStartCapture'
        pass

    def do_acquisition(self):
        value = self.alazar.acquire(**self.acquisitionkwargs)
        return value

    def handle_buffer(self, data):
        self.buffer += data

    def post_acquire(self):
        # average over records in buffer:
        # for ATS9360 samples are arranged in the buffer as follows:
        # S0A, S0B, ..., S1A, S1B, ...
        # with SXY the sample number X of channel Y.
        records_per_acquisition = (1. * self.buffers_per_acquisition *
                                   self.records_per_buffer)

        if self.average_mode() == 'none':
            raise NameError('Not implemented yet')
        elif self.average_mode() == 'trace':
            records = [np.zeros(self.samples_per_record) for k in range(self.number_of_channels)]

            for channel in range(self.number_of_channels):
                channel_offset = channel * self.samples_per_record * self.records_per_buffer
                for i in range(self.records_per_buffer):
                    i0 = channel_offset + i * self.samples_per_record
                    i1 = i0 + self.samples_per_record
                    records[channel] += self.buffer[i0:i1] / records_per_acquisition
        elif self.average_mode() == 'point':
            trace_length = self.samples_per_record * self.records_per_buffer
            records = [np.mean(self.buffer[i*trace_length:(i+1)*trace_length])/ records_per_acquisition
                       for i in range(self.number_of_channels)]

        # Scale datapoints
        for i, record in enumerate(records):
            channel_range = eval('self.alazar.channel_range{}()'.format(i + 1))
            # Somehow if buffers_per_acquisition=1, a different offset is needed
            if self.buffers_per_acquisition == 1:
                records[i] = 2 * (record / 2 ** 16 - 1) * channel_range
            else:
                records[i] = 2 * (record / 2 ** 16 - 0.5) * channel_range
        return records


# DFT AcquisitionController
class DFT_AcquisitionController(AcquisitionController):
    def __init__(self, name, alazar_id, demodulation_frequency, **kwargs):
        self.demodulation_frequency = demodulation_frequency
        self.acquisitionkwargs = {'acquisition_controller': self}
        self.samples_per_record = None
        self.bits_per_sample = None
        self.records_per_buffer = None
        self.buffers_per_acquisition = None
        # TODO(damazter) (S) this is not very general:
        self.number_of_channels = 2
        self.cos_list = None
        self.sin_list = None
        self.buffer = None
        # make a call to the parent class and by extension, create the parameter
        # structure of this class
        super().__init__(name, alazar_id, **kwargs)
        self.add_parameter("acquisition", get_cmd=self.do_acquisition)

    def set_acquisitionkwargs(self, **kwargs):
        self.acquisitionkwargs.update(**kwargs)

    def do_acquisition(self):
        value = self.get_alazar().acquire(**self.acquisitionkwargs)
        return value

    def pre_start_capture(self):
        alazar = self.get_alazar()
        self.samples_per_record = alazar.samples_per_record.get()
        self.records_per_buffer = alazar.records_per_buffer.get()
        self.buffers_per_acquisition = alazar.buffers_per_acquisition.get()
        sample_speed = alazar.get_sample_rate()
        integer_list = np.arange(self.samples_per_record)
        angle_list = (2 * np.pi * self.demodulation_frequency / sample_speed *
                      integer_list)

        self.cos_list = np.cos(angle_list)
        self.sin_list = np.sin(angle_list)
        self.buffer = np.zeros(self.samples_per_record *
                               self.records_per_buffer *
                               self.number_of_channels)

    def pre_acquire(self):
        # this could be used to start an Arbitrary Waveform Generator, etc...
        # using this method ensures that the contents are executed AFTER the
        # Alazar card starts listening for a trigger pulse
        pass

    def handle_buffer(self, data):
        self.buffer += data

    def post_acquire(self):
        alazar = self.get_alazar()
        # average all records in a buffer
        records_per_acquisition = (1. * self.buffers_per_acquisition *
                                   self.records_per_buffer)
        recordA = np.zeros(self.samples_per_record)
        for i in range(self.records_per_buffer):
            i0 = i * self.samples_per_record
            i1 = i0 + self.samples_per_record
            recordA += self.buffer[i0:i1] / records_per_acquisition

        recordB = np.zeros(self.samples_per_record)
        for i in range(self.records_per_buffer):
            i0 = i * self.samples_per_record + len(self.buffer) / 2
            i1 = i0 + self.samples_per_record
            recordB += self.buffer[i0:i1] / records_per_acquisition

        if self.number_of_channels == 2:
            # fit channel A and channel B
            res1 = self.fit(recordA)
            res2 = self.fit(recordB)
            return [alazar.signal_to_volt(1, res1[0] + 127.5),
                    alazar.signal_to_volt(2, res2[0] + 127.5),
                    res1[1], res2[1],
                    (res1[1] - res2[1]) % 360]
        else:
            raise Exception("Could not find CHANNEL_B during data extraction")
        return None

    def fit(self, buf):
        # Discrete Fourier Transform
        RePart = np.dot(buf - 127.5, self.cos_list) / self.samples_per_record
        ImPart = np.dot(buf - 127.5, self.sin_list) / self.samples_per_record

        # the factor of 2 in the amplitude is due to the fact that there is
        # a negative frequency as well
        ampl = 2 * np.sqrt(RePart ** 2 + ImPart ** 2)

        # see manual page 52!!! (using unsigned data)
        return [ampl, math.atan2(ImPart, RePart) * 360 / (2 * math.pi)]
