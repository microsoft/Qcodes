import math
import numpy as np
import logging
import time

from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals
from .ATS import AcquisitionController


class Triggered_AcquisitionController(AcquisitionController):
    """Basic AcquisitionController tested on ATS9360
    returns unprocessed data averaged by record with 2 channels
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
                self.buffers[ch][buffer_slice] = segmented_buffer[ch_name]
        else:
            print('*'*20+'\nIgnoring extra ATS buffer')
            pass
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


class SteeredInitialization_AcquisitionController(Continuous_AcquisitionController):
    shared_kwargs = ['target_instrument']

    def __init__(self, name, alazar_name, target_instrument, **kwargs):
        super().__init__(name, alazar_name, **kwargs)

        self._target_instrument = target_instrument

        self.add_parameter(name='t_no_blip',
                           parameter_class=ManualParameter,
                           units='ms',
                           initial_value=40)
        self.add_parameter(name='t_max_wait',
                           parameter_class=ManualParameter,
                           units='ms',
                           initial_value=500)
        self.add_parameter(name='max_wait_action',
                           parameter_class=ManualParameter,
                           units='ms',
                           initial_value='start',
                           vals=vals.Enum('start', 'error'))

        self.add_parameter(name='silent',
                           parameter_class=ManualParameter,
                           initial_value=True,
                           vals=vals.Bool())
        self.add_parameter(name='stage',
                           parameter_class=ManualParameter,
                           vals=vals.Enum('initialization', 'active', 'read'))
        self.add_parameter(name='record_initialization_traces',
                           parameter_class=ManualParameter,
                           initial_value=False,
                           vals=vals.Bool())

        # Parameters for the target instrument and command after initialization
        self.add_parameter(name='target_instrument',
                           get_cmd=lambda: self._target_instrument.name)
        self.add_parameter(name='target_command',
                           parameter_class=ManualParameter,
                           initial_value='start',
                           vals=vals.Strings())


        self.add_parameter(name='readout_channel',
                           parameter_class=ManualParameter,
                           vals=vals.Enum('A', 'B', 'C', 'D'))
        self.add_parameter(name='trigger_channel',
                           parameter_class=ManualParameter,
                           vals=vals.Enum('A', 'B', 'C', 'D'))
        self.add_parameter(name='trigger_threshold_voltage',
                           parameter_class=ManualParameter)
        self.add_parameter(name='readout_threshold_voltage',
                           parameter_class=ManualParameter)

        self.add_parameter(name='initialization_traces',
                           get_cmd=lambda: self._initialization_traces)
        self.add_parameter(name='post_initialization_traces',
                           get_cmd=lambda: self._post_initialization_traces)

        self.number_of_buffers_no_blip = None
        self.number_of_buffers_max_wait = None
        self._target_command = None
        self.buffer_no_blip_idx = None
        self.trace_idx = None
        self.t_start_list = None

    def setup(self, readout_threshold_voltage=None,
              trigger_threshold_voltage=None, **kwargs):
        if readout_threshold_voltage is not None:
            self.readout_threshold_voltage(readout_threshold_voltage)
        if trigger_threshold_voltage is not None:
            self.trigger_threshold_voltage(trigger_threshold_voltage)
        super().setup()

        # Convert t_no_blip and t_max_wait to equivalent number of buffers
        sample_rate = self._alazar.get_sample_rate()
        samples_per_ms = sample_rate * 1e-3
        buffers_per_ms = samples_per_ms / self.samples_per_record
        self.ms_per_buffer = 1 / buffers_per_ms
        self.number_of_buffers_max_wait = \
            int(np.ceil(self.t_max_wait() * buffers_per_ms))
        self.number_of_buffers_no_blip = \
            int(np.ceil(self.t_no_blip() * buffers_per_ms))

        # Setup target command when t_no_blip is reached
        self._target_command = getattr(self._target_instrument,
                                       self.target_command())

        self.t_start_list = np.zeros(self.traces_per_acquisition())

        if self.record_initialization_traces():
            self._initialization_traces = np.zeros(
                (self.traces_per_acquisition(),
                 self.samples_per_record * self.number_of_buffers_max_wait))
            self._post_initialization_traces = {
                ch_idx: np.zeros((self.traces_per_acquisition(),
                                  self.samples_per_record))
                for ch_idx in self.channel_selection}

    def pre_start_capture(self):
        """
        Initializes buffers before capturing
        """
        self.t0 = time.time()

        super().pre_start_capture()
        self.stage('initialization')
        self.buffer_no_blip_idx = 0

    def handle_buffer(self, buffer):
        if self.stage() == 'initialization':
            self.buffer_idx += 1
            # increase no_blip_idx, if there was a blip it will be reset
            self.buffer_no_blip_idx += 1

            segmented_buffers = self.segment_buffer(buffer, scale_voltages=True)
            readout_buffer = segmented_buffers[self.readout_channel()]

            if self.record_initialization_traces():
                # Store buffer into initialization trace
                buffer_slice = (
                    self.trace_idx, slice(
                        (self.buffer_idx-1) * self.samples_per_record,
                        self.buffer_idx * self.samples_per_record))
                self._initialization_traces[buffer_slice] = readout_buffer

            if max(readout_buffer) > self.readout_threshold_voltage():
                # A blip occurred, reset counter for buffers without blips
                self.buffer_no_blip_idx = 0

            if self.buffer_no_blip_idx >= self.number_of_buffers_no_blip:
                # Sufficient successive buffers without blips, starting
                self.stage('active')
                # Perform target command (e.g. starting an instrument)
                self._target_command()
                self.t_start_list[self.trace_idx] = self.buffer_idx * \
                                                    self.ms_per_buffer
                if not self.silent():
                    print('Starting active {:.2f} s'.format(
                        time.time() - self.t0))
                self.buffer_idx = 0
            elif self.buffer_idx >= self.number_of_buffers_max_wait:
                # Max waiting time has been reached, but no sufficient
                # period of time without blips occurred. Either start or error
                if self.max_wait_action() == 'start':
                    self.stage('active')
                    self._target_command()
                    self.t_start_list[self.trace_idx] = self.buffer_idx * \
                                                        self.ms_per_buffer
                    print('Starting active {:.2f} s'.format(time.time()-self.t0))
                    self.buffer_idx = 0

                    if not self.silent():
                        logging.warning('Max wait time reached, but no '
                                        'period without blips occurred, '
                                        'starting')
                else: # self.max_wait_action() == 'error':
                    raise RuntimeError('Max wait time reached but no period'
                                       ' without blips occurred, stopping')

        elif self.stage() == 'active':
            # Keep acquiring buffers until acquisition trigger is measured
            segmented_buffers = self.segment_buffer(buffer, scale_voltages=True)

            if self.buffer_idx == 0 and self.record_initialization_traces():
                # Store first stage after initialization
                for ch_idx in self.channel_selection:
                    self._post_initialization_traces[ch_idx][self.trace_idx] = \
                        segmented_buffers[ch_idx]

            trigger_buffer = segmented_buffers[self.trigger_channel()]
            if max(trigger_buffer) > self.trigger_threshold_voltage():
                # Acquisition trigger measured
                self.stage('read')

                # Find first index of acquisition trigger
                self.buffer_idx = -1
                self.buffer_start_idx = \
                    np.argmax(trigger_buffer > self.trigger_threshold_voltage())

                # Add first (partial) segment to traces
                super().handle_buffer(buffer)

                if not self.silent():
                    print('starting readout after {:.2f} s, buffers {}'.format(
                        time.time()-self.t0, self.buffer_idx))
            else:
                self.buffer_idx += 1


        elif self.stage() == 'read':
            # Add buffer to data
            super().handle_buffer(buffer)
            if not self.buffer_idx % self.buffers_per_trace:

                if not self.silent():
                    print('finished trace after {:.2f} s'.format(
                        time.time()-self.t0))
                self.t0 = time.time()
                # Trace filled, go back to initialization
                self.stage('initialization')
                # Reset buffer idx
                self.buffer_idx = 0
                self.buffer_start_idx
                self.buffer_no_blip_idx = 0

        else:
            raise ValueError('Acquisition stage {} unknown'.format(self.stage))


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
