from .ATS import AcquisitionController
import math
import numpy as np
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals
import inspect


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

        self.acquisitionkwargs = {}
        # Obtain a list of all valid ATS acquisition kwargs
        self._acquisitionkwargs_names = inspect.signature(self.alazar.acquire).parameters.keys()

        self.add_parameter(name='average_mode',
                           parameter_class=ManualParameter,
                           initial_value='trace',
                           vals=vals.Enum('none', 'trace', 'point'))
        # Names and shapes must have initial value, even through they will be overwritten in set_acquisitionkwargs.
        # If we don't do this, the remoteInstrument will not recognize that it returns multiple values.
        self.add_parameter(name="acquisition",
                           names=['channel_signal'],
                           get_cmd=self.do_acquisition,
                           shapes=((),),
                           snapshot_value=False)

    def get_acquisition_kwarg(self, kwarg):
        """
        Obtain an acquisition kwarg for the ATS.
        It first checks if the kwarg is an actual ATS acquisition kwarg, and raises an error otherwise.
        It then checks if the kwarg is in ATS_controller._acquisitionkwargs.
        If not, it will retrieve the ATS latest parameter value

        Args:
            kwarg: acquisition kwarg to look for

        Returns:
            Value of the acquisition kwarg
        """
        assert kwarg in self._acquisitionkwargs_names, \
            "Kwarg {} is not a valid ATS acquisition kwarg".format(kwarg)
        if kwarg in self.acquisitionkwargs.keys():
            return self.acquisitionkwargs[kwarg]
        else:
            # Must get latest value, since it may not be updated in ATS
            return self.alazar.parameters[kwarg].get_latest()

    def update_acquisitionkwargs(self, **kwargs):
        self.acquisitionkwargs.update(**kwargs)

        # Update acquisition parameter values. These depend on the average mode
        channel_selection = self.get_acquisition_kwarg('channel_selection')
        samples_per_record = self.get_acquisition_kwarg('samples_per_record')
        records_per_buffer = self.get_acquisition_kwarg('records_per_buffer')
        buffers_per_acquisition = self.get_acquisition_kwarg('buffers_per_acquisition')
        self.acquisition.names = tuple(['Channel_{}_signal'.format(ch) for ch in
                                        self.get_acquisition_kwarg('channel_selection')])

        self.acquisition.labels = self.acquisition.names
        self.acquisition.units = ['V'*len(channel_selection)]

        if self.average_mode() == 'point':
            self.acquisition.shapes = tuple([()]*len(channel_selection))
        elif self.average_mode() == 'trace':
            shape = (samples_per_record,)
            self.acquisition.shapes = tuple([shape] * len(channel_selection))
        else:
            shape = (records_per_buffer * buffers_per_acquisition, samples_per_record)
            self.acquisition.shapes = tuple([shape] * len(channel_selection))

    def pre_start_capture(self):
        self.samples_per_record = self.alazar.samples_per_record()
        self.records_per_buffer = self.alazar.records_per_buffer()
        self.buffers_per_acquisition = self.alazar.buffers_per_acquisition()
        self.number_of_channels = len(self.alazar.channel_selection())
        self.buffer_idx = 0
        if self.average_mode() in ['point', 'trace']:
            self.buffer = np.zeros(self.samples_per_record *
                                  self.records_per_buffer *
                                  self.number_of_channels)
        else:
            self.buffer = np.zeros((self.buffers_per_acquisition,
                                    self.samples_per_record *
                                    self.records_per_buffer *
                                    self.number_of_channels))

    def pre_acquire(self):
        # gets called after 'AlazarStartCapture'
        pass

    def do_acquisition(self):
        records = self.alazar.acquire(acquisition_controller=self,
                                      **self.acquisitionkwargs)
        return records

    def handle_buffer(self, data):
        print('ADDING BUFFER')
        if self.buffer_idx < self.buffers_per_acquisition:
            if self.average_mode() in ['point', 'trace']:
                self.buffer += data
            else:
                    self.buffer[self.buffer_idx] = data
        else:
            pass
            # print('*'*20+'\nIgnoring extra ATS buffer')
        self.buffer_idx += 1

    def post_acquire(self):
        # average over records in buffer:
        # for ATS9360 samples are arranged in the buffer as follows:
        # S0A, S0B, ..., S1A, S1B, ...
        # with SXY the sample number X of channel Y.
        records_per_acquisition = self.buffers_per_acquisition * self.records_per_buffer
        channel_offset = lambda channel: channel * self.samples_per_record * self.records_per_buffer

        if self.average_mode() == 'none':
            records = [self.buffer[:, channel_offset(ch):channel_offset(ch+1)
                                  ].reshape((records_per_acquisition,
                                             self.samples_per_record))
                       for ch in range(self.number_of_channels)]
        elif self.average_mode() == 'trace':
            records = [np.zeros(self.samples_per_record) for k in range(self.number_of_channels)]

            for channel in range(self.number_of_channels):
                for i in range(self.records_per_buffer):
                    i0 = channel_offset(channel) + i * self.samples_per_record
                    i1 = i0 + self.samples_per_record
                    records[channel] += self.buffer[i0:i1] / records_per_acquisition
        elif self.average_mode() == 'point':
            trace_length = self.samples_per_record * self.records_per_buffer
            records = [np.mean(self.buffer[i*trace_length:(i+1)*trace_length])/ records_per_acquisition
                       for i in range(self.number_of_channels)]

        # Scale datapoints
        for i, record in enumerate(records):
            channel_range = eval('self.alazar.channel_range{}()'.format(i + 1))
            records[i] = 2 * (record / 2 ** 16 - 0.5) * channel_range
        return records


# DFT AcquisitionController
class Demodulation_AcquisitionController(AcquisitionController):
    """
    This class represents an example acquisition controller. End users will
    probably want to use something more sophisticated. It will average all
    buffers and then perform a fourier transform on the resulting average trace
    for one frequency component. The amplitude of the result of channel_a will
    be returned.

    args:
    name: name for this acquisition_conroller as an instrument
    alazar_name: the name of the alazar instrument such that this controller
        can communicate with the Alazar
    demodulation_frequency: the selected component for the fourier transform
    **kwargs: kwargs are forwarded to the Instrument base class
    """
    def __init__(self, name, alazar_name, demodulation_frequency, **kwargs):
        self.demodulation_frequency = demodulation_frequency
        self.acquisitionkwargs = {}
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

    def update_acquisitionkwargs(self, **kwargs):
        """
        This method must be used to update the kwargs used for the acquisition
        with the alazar_driver.acquire
        :param kwargs:
        :return:
        """
        self.acquisitionkwargs.update(**kwargs)

    def do_acquisition(self):
        """
        this method performs an acquisition, which is the get_cmd for the
        acquisiion parameter of this instrument
        :return:
        """
        value = self._get_alazar().acquire(acquisition_controller=self,
                                           **self.acquisitionkwargs)
        return value

    def pre_start_capture(self):
        """
        See AcquisitionController
        :return:
        """
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
        """
        See AcquisitionController
        :return:
        """
        # this could be used to start an Arbitrary Waveform Generator, etc...
        # using this method ensures that the contents are executed AFTER the
        # Alazar card starts listening for a trigger pulse
        pass

    def handle_buffer(self, data):
        """
        See AcquisitionController
        :return:
        """
        self.buffer += data

    def post_acquire(self):
        """
        See AcquisitionController
        :return:
        """
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
            i0 = i * self.samples_per_record + len(self.buffer) // 2
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
        """
        the DFT is implemented in this method
        :param buf: buffer to perform the transform on
        :return: return amplitude and phase of the resulted transform
        """
        # Discrete Fourier Transform
        RePart = np.dot(buf - 127.5, self.cos_list) / self.samples_per_record
        ImPart = np.dot(buf - 127.5, self.sin_list) / self.samples_per_record

        # the factor of 2 in the amplitude is due to the fact that there is
        # a negative frequency as well
        ampl = 2 * np.sqrt(RePart ** 2 + ImPart ** 2)

        # see manual page 52!!! (using unsigned data)
        return [ampl, math.atan2(ImPart, RePart) * 360 / (2 * math.pi)]
