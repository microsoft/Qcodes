import math
import numpy as np
import logging
import time

from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals
from .ATS import AcquisitionController


class Triggered_AcquisitionController(AcquisitionController):
    """
    Acquisition controller that acquires a record after each trigger.

    The resulting data is a list, where each element corresponds to an ATS
    acquisition channel, and whose shape depends on post-processing.
    The parameter average_mode sets the data averaging during post-processing.
    Possible modes are:
        'none': Return full traces, each output data element has shape
                    (records_per_buffer * buffers_per_acquisition,
                     samples_per_record)
        'trace': Average over all traces (records), with data element shape
                    (samples_per_record)
        'point': Average over all traces and over time, returning the average
                 signal as a single value.
    """
    def __init__(self, name, alazar_name, **kwargs):
        super().__init__(name, alazar_name, **kwargs)
        self.samples_per_record = None
        self.records_per_buffer = None
        self.buffers_per_acquisition = None
        self.buffer = None
        self.buffer_idx = 0

        self._fixed_acquisition_settings = {
            'mode': 'NPT'
        }

        self.add_parameter(name='average_mode',
                           parameter_class=ManualParameter,
                           initial_value='trace',
                           vals=vals.Enum('none', 'trace', 'point'))

    def setup(self, **kwargs):
        """
        Setup the ATS controller by updating most current ATS values and setting
        the acquisition parameter metadata

        Returns:
            None
        """
        # Update acquisition parameter values. These depend on the average mode
        for attr in ['channel_selection', 'samples_per_record',
                     'records_per_buffer', 'buffers_per_acquisition']:
            setattr(self, attr, self.get_acquisition_setting(attr))
        self.samples_per_buffer = self.samples_per_record * \
                                  self.records_per_buffer
        self.number_of_channels = len(self.channel_selection)
        self.traces_per_acquisition = self.buffers_per_acquisition * \
                                       self.records_per_buffer

        if self.samples_per_record % 16:
            raise SyntaxError('Samples per record {} is not multiple of '
                              '16'.format(self.samples_per_record))

        # Set acquisition parameter metadata
        self.acquisition.names = tuple(['ch{}_signal'.format(ch) for ch in
                                        self.channel_selection])
        self.acquisition.labels = self.acquisition.names
        self.acquisition.units = ['V'] * self.number_of_channels

        if self.average_mode() == 'point':
            shape = ()
        elif self.average_mode() == 'trace':
            shape = (self.samples_per_record,)
        else:
            shape = (self.traces_per_acquisition, self.samples_per_record)
        self.acquisition.shapes = tuple([shape] * self.number_of_channels)

    def _requires_buffer(self):
        return self.buffer_idx < self.buffers_per_acquisition

    def pre_start_capture(self):
        """
        Initializes buffers before capturing
        """
        self.buffer_idx = 0
        self.buffers = [np.zeros((self.traces_per_acquisition,
                                  self.samples_per_record))
                        for ch in self.channel_selection]

    def pre_acquire(self):
        # gets called after 'AlazarStartCapture'
        pass

    def handle_buffer(self, buffer):
        if self.buffer_idx < self.buffers_per_acquisition:
            # Segment the buffer into buffers for each channel
            segmented_buffer = self.segment_buffer(buffer, scale_voltages=True)

            # Save buffer components into each channel dataset
            for ch, ch_name in enumerate(self.channel_selection):
                buffer_slice = slice(
                    self.buffer_idx * self.records_per_buffer,
                    (self.buffer_idx + 1) * self.records_per_buffer)
                self.buffers[ch][buffer_slice] = segmented_buffer.pop(ch_name)
        else:
            logging.warning('Ignoring extra ATS buffer')

        self.buffer_idx += 1

    def post_acquire(self):
        # average over records in buffer:
        # for ATS9360 samples are arranged in the buffer as follows:
        # S0A, S0B, ..., S1A, S1B, ...
        # with SXY the sample number X of channel Y.

        if self.average_mode() == 'none':
            data = self.buffers
        elif self.average_mode() == 'trace':
            data = [np.mean(buffer, axis=0) for buffer in self.buffers]
        elif self.average_mode() == 'point':
            data = [np.mean(buffer) for buffer in self.buffers]

        return data


class Continuous_AcquisitionController(AcquisitionController):
    """
    Acquisition controller that continuously acquires data without needing a
    trigger. In contrast to triggered mode, here a trace may be composed of
    multiple buffers. Therefore, the acquisition settings 'records_per_buffer'
    and 'buffers_per_acquisition' are fixed, and instead the parameters
    'samples_per_trace' and 'traces_per_acquisition' must be set.

    The resulting data is a list, where each element corresponds to an ATS
    acquisition channel, and whose shape depends on post-processing.
    The parameter average_mode sets the data averaging during post-processing.
    Possible modes are:
        'none': Return full traces, each output data element has shape
                    (traces_per_acquisition,
                     samples_per_trace)
        'trace': Average over all traces (records), with data element shape
                    (samples_per_trace)
        'point': Average over all traces and over time, returning the average
                 signal as a single value.
    """
    def __init__(self, name, alazar_name, **kwargs):
        super().__init__(name, alazar_name, **kwargs)

        # buffers_per_acquisition=0x7FFFFFFF results in buffers being collected
        # indefinitely until aborted.
        # Records_per_buffer must equal 1 for CS or TS
        self._fixed_acquisition_settings = {
            'mode': 'CS',
            'buffers_per_acquisition': 0x7FFFFFFF,
            'records_per_buffer': 1}
        self._acquisition_settings = self._fixed_acquisition_settings.copy()

        self.add_parameter(name='average_mode',
                           parameter_class=ManualParameter,
                           initial_value='trace',
                           vals=vals.Enum('none', 'trace', 'point'))
        self.add_parameter(name='samples_per_trace',
                           parameter_class=ManualParameter,
                           vals=vals.Multiples(divisor=16))
        self.add_parameter(name='traces_per_acquisition',
                           parameter_class=ManualParameter,
                           vals=vals.Ints())

        self.buffer_idx = None
        self.trace_idx = None
        self.buffer_start_idx = None

    def setup(self, **kwargs):
        """
        Setup the ATS controller by updating most current ATS values and setting
        the acquisition parameter metadata

        Returns:
            None
        """

        # Update acquisition parameter values. These depend on the average mode
        for attr in ['channel_selection', 'samples_per_record']:
            setattr(self, attr, self.get_acquisition_setting(attr))
        self.number_of_channels = len(self.channel_selection)
        self.buffers_per_trace = round(self.samples_per_trace() / \
                                       self.samples_per_record)

        if self.samples_per_record % 16:
            raise SyntaxError('Samples per record {} is not multiple of '
                              '16'.format(self.samples_per_record))

        # Set acquisition parameter metadata
        self.acquisition.names = tuple(['ch{}_signal'.format(ch) for ch in
                                        self.channel_selection])
        self.acquisition.labels = self.acquisition.names
        self.acquisition.units = ['V'] * self.number_of_channels

        if self.average_mode() == 'point':
            shape = ()
        elif self.average_mode() == 'trace':
            shape = (self.samples_per_trace(),)
        else:
            shape = (self.traces_per_acquisition(), self.samples_per_record)
        self.acquisition.shapes = tuple([shape] * self.number_of_channels)

    def _requires_buffer(self):
        return self.trace_idx < self.traces_per_acquisition()

    def pre_start_capture(self):
        """
        Initializes buffers before capturing
        """
        self.buffer_idx = 0
        self.trace_idx = 0
        self.buffer_start_idx = 0
        self.buffers = [np.zeros((self.traces_per_acquisition(),
                                  self.samples_per_trace()))
                        for ch in self.channel_selection]

    def pre_acquire(self):
        # gets called after 'AlazarStartCapture'
        pass

    def handle_buffer(self, buffer):
        """
        Adds buffers to fill up the data traces.
        Because a trace can consist of multiple buffers, the 'trace_idx' and
        'buffer_idx' determine the relevant trace/buffer within a trace,
        respectively.
        Note that 'buffer_start_idx' set to nonzero if the first buffer should
        be added from a nonzero starting idx. In this case, the first buffer
        must have 'buffer_idx = -1'.
        Args:
            buffer: Buffer to add to trace

        Returns:
            None
        """
        if self.buffer_idx >= self.buffers_per_trace * \
                             self.traces_per_acquisition():
            print('Ignoring extra ATS buffer {}'.format(self.buffer_idx))
            return None

        # Segment the buffer into buffers for each channel
        segmented_buffer = self.segment_buffer(buffer, scale_voltages=True)

        # Determine the slice where the buffer segments should go in the traces,
        # and possibly cut the segments if they start at nonzero idx,
        # or if they don't fit completely in a trace
        if self.buffer_idx == -1:
            # This is the first (incomplete) buffer, whose starting idx is
            # buffer_start_idx, add it to the beginning of the segmented buffer
            buffer_slice = slice(None, self.samples_per_record -
                                 self.buffer_start_idx)
            for ch_name, buffer_segment in segmented_buffer.items():
                # Shorten each of the segments to start from buffer_start_idx
                segmented_buffer[ch_name] = \
                    buffer_segment[self.buffer_start_idx:]
        else:
            # Determine the buffer idx offset in the trace
            trace_offset = self.samples_per_record - self.buffer_start_idx + \
                           self.samples_per_record * \
                           (self.buffer_idx % self.buffers_per_trace)

            if trace_offset+self.samples_per_record > self.samples_per_trace():
                # Buffer does not fit completely in a trace, cutting buffer
                buffer_slice = slice(trace_offset, None)
                for ch_name, buffer_segment in segmented_buffer.items():
                    # Shorten each of the segments to stop at samples_per_trace
                    max_idx = self.samples_per_trace() - trace_offset
                    segmented_buffer[ch_name] = buffer_segment[:max_idx]
            else:
               buffer_slice = slice(trace_offset,
                                    trace_offset + self.samples_per_record)

        # Save buffer components into each channel dataset
        for ch, ch_name in enumerate(self.channel_selection):
            self.buffers[ch][self.trace_idx, buffer_slice] = \
                segmented_buffer[ch_name]

        self.buffer_idx += 1
        if self.buffer_idx and not self.buffer_idx % self.buffers_per_trace:
            # Filled a trace
            self.trace_idx += 1

    def post_acquire(self):
        # average over records in buffer:
        # for ATS9360 samples are arranged in the buffer as follows:
        # S0A, S0B, ..., S1A, S1B, ...
        # with SXY the sample number X of channel Y.

        if self.average_mode() == 'none':
            data = self.buffers
        elif self.average_mode() == 'trace':
            data = [np.mean(buffer, axis=0) for buffer in self.buffers]
        elif self.average_mode() == 'point':
            data = [np.mean(buffer) for buffer in self.buffers]

        return data


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
