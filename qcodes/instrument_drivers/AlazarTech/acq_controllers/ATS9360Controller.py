import logging

import numpy as np
import logging

import qcodes.instrument_drivers.AlazarTech.acq_helpers as helpers
from qcodes import ChannelList
from qcodes import ManualParameter
from qcodes.instrument_drivers.AlazarTech.acq_controllers.alazar_channel import AlazarChannel
from ..ATS import AcquisitionController
from ..acquisition_parameters import AcqVariablesParam, \
    NonSettableDerivedParameter
from .demodulator import Demodulator

logger = logging.getLogger(__name__)

class ATS9360Controller(AcquisitionController):
    """
    This is the Acquisition Controller class which works with the ATS9360,
    averaging over records and buffers and demodulating with software reference
    signal(s). It may optionally integrate over the samples following the post processing


    Args:
        name: name for this acquisition_controller as an instrument
        alazar_name: name of the alazar instrument such that this
            controller can communicate with the Alazar
        filter (default 'win'): filter to be used to filter out double freq
            component ('win' - window, 'ls' - least squared, 'ave' - averaging)
        numtaps (default 101): number of freq components used in filter
        **kwargs: kwargs are forwarded to the Instrument base class

    TODO(nataliejpg) test filter options
    TODO(JHN) Use filtfit for better performance?
    TODO(JHN) Test demod+filtering and make it more modular
    TODO(nataliejpg) finish implementation of channel b option
    TODO(JHN) Option to not read channel b at all (Speedup)
    TODO(nataliejpg) what should be private?
    TODO(nataliejpg) where should filter_dict live?
    """

    filter_dict = {'win': 0, 'ls': 1, 'ave': 2}

    def __init__(self, name, alazar_name, filter: str = 'win',
                 numtaps: int =101,
                 **kwargs):
        super().__init__(name, alazar_name, **kwargs)
        self.filter_settings = {'filter': self.filter_dict[filter],
                                'numtaps': numtaps}
        self.number_of_channels = 2

        channels = ChannelList(self, "Channels", AlazarChannel)
        self.add_submodule("channels", channels)

        self.add_parameter(name='int_time',
                           check_and_update_fn=self._update_int_time,
                           default_fn=self._int_time_default,
                           parameter_class=AcqVariablesParam)
        self.add_parameter(name='int_delay',
                           check_and_update_fn=self._update_int_delay,
                           default_fn=self._int_delay_default,
                           parameter_class=AcqVariablesParam)
        self.add_parameter(name='allocated_buffers',
                           alternative='not controllable in this controller',
                           parameter_class=NonSettableDerivedParameter)
        self.add_parameter(name='samples_per_record',
                           alternative='int_time and int_delay',
                           parameter_class=NonSettableDerivedParameter)

        self.samples_divisor = self._get_alazar().samples_divisor

        self.active_channels = []

    def _update_int_time(self, value, **kwargs):
        """
        Function to validate value for int_time before setting parameter
        value and update instr attributes.

        Args:
            value to be validated and used for instrument attribute update

        Checks:
            0 <= value <= 0.1 seconds
            number of oscillation measured in this time
            oversampling rate

        Sets:
            samples_per_record of acq controller to match int_time and int_delay
        """
        if (value is None) or not (0 <= value <= 0.1):
            raise ValueError('int_time must be 0 <= value <= 1')

        alazar = self._get_alazar()
        sample_rate = alazar.effective_sample_rate.get()
        max_demod_freq = 0
        for channel in self.channels:
            if channel._demod:
                demod_freq = channel.demod_freq()
                max_demod_freq = max(max_demod_freq, demod_freq)
        if max_demod_freq > 0:
            Demodulator.verify_demod_freq(max_demod_freq, sample_rate, value)
        if self.int_delay() is None:
            self.int_delay.to_default()
        int_delay = self.int_delay.get()
        self._update_samples_per_record(sample_rate, value, int_delay)

    def _update_samples_per_record(self, sample_rate, int_time, int_delay):
        """
        Keeps non settable samples_per_record up to date with int_time int_delay
        and updates setpoints as needed.
        """
        total_time = (int_time or 0) + (int_delay or 0)
        samples_needed = total_time * sample_rate
        samples_per_record = helpers.roundup(
            samples_needed, self.samples_divisor)
        logger.info("need {} samples round up to {}".format(samples_needed, samples_per_record))
        self.samples_per_record._save_val(samples_per_record)

    def _update_int_delay(self, value, **kwargs):
        """
        Function to validate value for int_delay before setting parameter
        value and update instr attributes.

        Args:
            value to be validated and used for instrument attribute update

        Checks:
            0 <= value <= 0.1 seconds
            number of samples discarded >= numtaps

        Sets:
            samples_per_record of acq controller to match int_time and int_delay
            samples_per_record of acq controller
            acquisition_kwarg['samples_per_record'] of acquisition param
            setpoints of acquisition param
        """
        int_delay_min = 0
        int_delay_max = 0.1
        if (value is None) or not (int_delay_min <= value <= int_delay_max):
            raise ValueError('int_delay must be {} <= value <= {}. Got {}'.format(int_delay_min,
                                                                                  int_delay_max,
                                                                                  value))
        alazar = self._get_alazar()
        sample_rate = alazar.effective_sample_rate.get()
        samples_delay_min = (self.filter_settings['numtaps'] - 1)
        int_delay_min = samples_delay_min / sample_rate
        if value < int_delay_min:
            logging.warning(
                'delay is less than recommended for filter choice: '
                '(expect delay >= {})'.format(int_delay_min))

        int_time = self.int_time.get()
        self._update_samples_per_record(sample_rate, int_time, value)

    def _int_delay_default(self):
        """
        Function to generate default int_delay value

        Returns:
            minimum int_delay recommended for (numtaps - 1)
            samples to be discarded as recommended for filter
        """
        alazar = self._get_alazar()
        sample_rate = alazar.effective_sample_rate.get()
        samp_delay = self.filter_settings['numtaps'] - 1
        return samp_delay / sample_rate

    def _int_time_default(self):
        """
        Function to generate default int_time value

        Returns:
            max total time for integration based on samples_per_record,
            sample_rate and int_delay
        """
        samples_per_record = self.samples_per_record.get()
        if samples_per_record in (0 or None):
            raise ValueError('Cannot set int_time to max if acq controller'
                             ' has 0 or None samples_per_record, choose a '
                             'value for int_time and samples_per_record will '
                             'be set accordingly')
        alazar = self._get_alazar()
        sample_rate = alazar.effective_sample_rate.get()
        total_time = ((samples_per_record / sample_rate) -
                      (self.int_delay() or 0))
        return total_time

    def update_filter_settings(self, filter, numtaps):
        """
        Updates the settings of the filter for filtering out
        double frequency component for demodulation.

        Args:
            filter (str): filter type ('win' or 'ls')
            numtaps (int): numtaps for filter
        """
        self.filter_settings.update({'filter': self.filter_dict[filter],
                                     'numtaps': numtaps})

    def update_acquisition_kwargs(self, **kwargs):
        """
        Updates the kwargs to be used when
        alazar_driver.acquisition() is called via a get call of the
        acquisition parameter. Should be used by the user for
        updating averaging settings. If integrating over samples
        'samples_per_record' cannot be set directly it should instead
        be set via the int_time and int_delay parameters.

        Kwargs (ints):
            - records_per_buffer
            - buffers_per_acquisition
            - allocated_buffers
        """
        if 'samples_per_record' in kwargs:
            raise ValueError('Samples_per_record should not be set manually '
                             'via update_acquisition_kwargs and should instead'
                             ' be set by setting int_time and int_delay')
        self.acquisition.acquisition_kwargs.update(**kwargs)

    def pre_start_capture(self):
        """
        Called before capture start to update Acquisition Controller with
        Alazar acquisition params and set up software wave for demodulation.
        """
        alazar = self._get_alazar()
        acq_s_p_r = self.samples_per_record.get()
        inst_s_p_r = alazar.samples_per_record.get()
        sample_rate = alazar.effective_sample_rate.get()
        if acq_s_p_r != inst_s_p_r:
            raise Exception('acq controller samples per record {} does not match'
                            ' instrument value {}, most likely need '
                            'to set and check int_time and int_delay'.format(acq_s_p_r, inst_s_p_r))

        samples_per_record = inst_s_p_r
        records_per_buffer = alazar.records_per_buffer.get()
        buffers_per_acquisition = alazar.buffers_per_acquisition.get()
        max_samples = self._get_alazar().get_idn()['max_samples']
        samples_per_buffer = records_per_buffer * samples_per_record
        if samples_per_buffer > max_samples:
            raise RuntimeError("Trying to acquire {} samples in one buffer maximum"
                               " supported is {}".format(samples_per_buffer, max_samples))

        self.board_info = alazar.get_idn()
        # extend this to support more than one channel
        settings = self.active_channels[0]
        if settings['average_buffers']:
            self.buffer = np.zeros(samples_per_record *
                                   records_per_buffer *
                                   self.number_of_channels)
        else:
            self.buffer = np.zeros((buffers_per_acquisition,
                                   samples_per_record *
                                   records_per_buffer *
                                   self.number_of_channels))

        if settings['demod']:
            self.demodulator = Demodulator(buffers_per_acquisition,
                                           records_per_buffer,
                                           samples_per_record,
                                           sample_rate,
                                           self.filter_settings,
                                           self.active_channels)

    def pre_acquire(self):
        pass

    def handle_buffer(self, data, buffernum=0):
        """
        Adds data from Alazar to buffer (effectively averaging)
        """
        settings = self.active_channels[0]
        if settings['average_buffers']:
            self.buffer += data
        else:
            self.buffer[buffernum] = data

    def post_acquire(self):
        """
        Processes the data according to ATS9360 settings, splitting into
        records and averaging over them, then applying demodulation fit
        nb: currently only channel A. Depending on the value of integrate_samples
        it may either sum over all samples or return arrays of individual samples
        for all the data given below.

        Returns:
            - Raw data
            - For each demodulation frequency:
                * magnitude
                * phase
        """

        # for ATS9360 samples are arranged in the buffer as follows:
        # S00A, S00B, S01A, S01B...S10A, S10B, S11A, S11B...
        # where SXYZ is record X, sample Y, channel Z.

        # break buffer up into records and averages over them
        alazar = self._get_alazar()
        samples_per_record = alazar.samples_per_record.get()
        records_per_buffer = alazar.records_per_buffer.get()
        buffers_per_acquisition = alazar.buffers_per_acquisition.get()
        settings = self.active_channels[0]
        if settings['average_buffers']:
            number_of_buffers = 1
        else:
            number_of_buffers = buffers_per_acquisition
        reshaped_buf = self.buffer.reshape(number_of_buffers,
                                           records_per_buffer,
                                           samples_per_record,
                                           self.number_of_channels)
        channel = settings['channel']
        channelData = reshaped_buf[..., channel]
        if settings['average_records'] and settings['average_buffers']:
            recordA = np.uint16(np.mean(channelData, axis=1, keepdims=True) /
                                buffers_per_acquisition)
        elif settings['average_records']:
            recordA = np.uint16(np.mean(channelData, axis=1, keepdims=True))
        elif settings['average_buffers']:
            recordA = np.uint16(channelData/buffers_per_acquisition)
        else:
            recordA = np.uint16(channelData)
        recordA = self._to_volts(recordA)

        # do demodulation
        if settings['demod']:
            magA, phaseA = self.demodulator.demodulate(recordA, self.int_delay(), self.int_time())
            data = magA[0]
        else:
            if settings['integrate_samples']:
                data = np.squeeze(np.mean(recordA, axis=-1))
            else:
                data = np.squeeze(recordA)

        return data

    def _to_volts(self, record):
        # convert rec to volts
        bps = self.board_info['bits_per_sample']
        if bps == 12:
            volt_rec = helpers.sample_to_volt_u12(record, bps, input_range_volts=0.4)
        else:
            logging.warning('sample to volt conversion does not exist for'
                            ' bps != 12, centered raw samples returned')
            volt_rec = record - np.mean(record)
        return volt_rec
