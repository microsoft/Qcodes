from .ATS import AcquisitionController
import math
import numpy as np
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals


class Basic_AcquisitionController(AcquisitionController):
    """Basic AcquisitionController tested on ATS9360
    returns unprocessed data averaged by record with 2 channels
    """
    def __init__(self, name, alazar_name, **kwargs):
        self.samples_per_record = None
        self.records_per_buffer = None
        self.buffers_per_acquisition = None
        self.buffer = None
        super().__init__(name, alazar_name, **kwargs)
        self.alazar = self._get_alazar()

        self.add_parameter(name='average_mode',
                           parameter_class=ManualParameter,
                           initial_value='trace',
                           vals=vals.Enum('none', 'trace', 'point'))


        self.buffer_idx = 0
        # Names and shapes must have initial value, even though they will be
        # overwritten in set_acquisitionkwargs. If we don't do this, the
        # RemoteInstrument will not recognize that it returns multiple values.
        self.add_parameter(name="acquisition",
                           names=['channel_signal'],
                           get_cmd=self.do_acquisition,
                           shapes=((),),
                           snapshot_value=False)

    def setup(self, **kwargs):
        """
        This function sets up the ATS for an acquisition.
        In particular, it updates the acquisition kwargs, and the attributes
        for the parameter acquisition.
        These attributes depend on the controller's average_mode.
        This function must be performed after setting the acquisitionkwargs,
        and before starting an actual Loop
        """

        self.update_acquisitionkwargs(**kwargs)

        channel_selection = self.get_acquisitionkwarg('channel_selection')
        samples_per_record = self.get_acquisitionkwarg('samples_per_record')
        records_per_buffer = self.get_acquisitionkwarg('records_per_buffer')
        buffers_per_acquisition = self.get_acquisitionkwarg(
            'buffers_per_acquisition')

        self.acquisition.names = tuple(
            ['Channel_{}_signal'.format(ch) for ch in channel_selection])

        self.acquisition.labels = self.acquisition.names
        self.acquisition.units = ['V'*len(channel_selection)]

        if self.average_mode() == 'point':
            self.acquisition.shapes = tuple([()]*len(channel_selection))
        elif self.average_mode() == 'trace':
            shape = (samples_per_record,)
            self.acquisition.shapes = tuple([shape] * len(channel_selection))
        else:
            shape = (records_per_buffer * buffers_per_acquisition,
                     samples_per_record)
            self.acquisition.shapes = tuple([shape] * len(channel_selection))

    def do_acquisition(self):
        value = self._get_alazar().acquire(acquisition_controller=self,
                                           **self.acquisitionkwargs())
        return value

    def pre_start_capture(self):
        alazar = self._get_alazar()
        number_of_channels = len(alazar.channel_selection.get())
        self.buffer_idx = 0
        if self.average_mode() in ['point', 'trace']:
            self.buffer = np.zeros(alazar.samples_per_record.get() *
                                   alazar.records_per_buffer.get() *
                                   number_of_channels)
        else:
            self.buffer = np.zeros((alazar.buffers_per_acquisition.get(),
                                    alazar.samples_per_record.get() *
                                    alazar.records_per_buffer.get() *
                                    number_of_channels))

    def pre_acquire(self):
        # gets called after 'AlazarStartCapture'
        pass

    def handle_buffer(self, data):
        if self.buffer_idx < self.buffers_per_acquisition:
            if self.average_mode() in ['point', 'trace']:
                self.buffer += data
            else:
                    self.buffer[self.buffer_idx] = data
        else:
            print('*'*20+'\nIgnoring extra ATS buffer')
        self.buffer_idx += 1

    def post_acquire(self):
        # Perform averaging over records.
        # The averaging mode depends on parameter average_mode
        records_per_acquisition = self.buffers_per_acquisition * \
                                  self.records_per_buffer

        def channel_offset(ch):
            return ch * self.samples_per_record * self.records_per_buffer

        if self.average_mode() == 'none':
            records = [self.buffer[:,
                       channel_offset(ch):channel_offset(ch+1)].reshape(
                (records_per_acquisition, self.samples_per_record))
                       for ch in range(self.number_of_channels)]
        elif self.average_mode() == 'trace':
            records = [np.zeros(self.samples_per_record) for _ in
                       range(self.number_of_channels)]

            for channel in range(self.number_of_channels):
                for i in range(self.records_per_buffer):
                    i0 = channel_offset(channel) + i * self.samples_per_record
                    i1 = i0 + self.samples_per_record
                    records[channel] += \
                        self.buffer[i0:i1] / records_per_acquisition
        elif self.average_mode() == 'point':
            trace_length = self.samples_per_record * self.records_per_buffer
            records = [np.mean(self.buffer[i*trace_length:(i+1)*trace_length]) /
                       records_per_acquisition
                       for i in range(self.number_of_channels)]

        # Scale datapoints
        for i, record in enumerate(records):
            channel_range = eval('self.alazar.channel_range{}()'.format(i + 1))
            records[i] = 2 * (record / 2 ** 16 - 0.5) * channel_range
        return records


# DFT AcquisitionController
class DFT_AcquisitionController(AcquisitionController):
    def __init__(self, name, alazar_name, demodulation_frequency, **kwargs):
        self.demodulation_frequency = demodulation_frequency
        self.samples_per_record = None
        self.records_per_buffer = None
        self.buffers_per_acquisition = None
        # TODO(damazter) (S) this is not very general:
        self.number_of_channels = 2
        self.cos_list = None
        self.sin_list = None
        self.buffer = None
        # make a call to the parent class and by extension, create the parameter
        # structure of this class
        super().__init__(name, alazar_name, **kwargs)
        self.add_parameter("acquisition", get_cmd=self.do_acquisition)

    def pre_start_capture(self):
        alazar = self._get_alazar()
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
        alazar = self._get_alazar()
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
            #return [alazar.signal_to_volt(1, res1[0] + 127.5),
            #        alazar.signal_to_volt(2, res2[0] + 127.5),
            #        res1[1], res2[1],
            #        (res1[1] - res2[1]) % 360]
            return alazar.signal_to_volt(1, res1[0] + 127.5)
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
