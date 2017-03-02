import logging
from ..ATS import AcquisitionController
import numpy as np
import qcodes.instrument_drivers.AlazarTech.acq_helpers as helpers
from ..acqusition_parameters import AcqVariablesParam, \
                                    ExpandingAlazarArrayMultiParameter, \
                                    NonSettableDerivedParameter, \
                                    DemodFreqParameter


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
        chan_b (default False): whether there is also a second channel of data
            to be processed and returned. Not currently fully implemented.
        **kwargs: kwargs are forwarded to the Instrument base class

    TODO(nataliejpg) test filter options
    TODO(nataliejpg) finish implementation of channel b option
    TODO(nataliejpg) what should be private?
    TODO(nataliejpg) where should filter_dict live?
    TODO(nataliejpg) demod_freq should be changeable number: maybe channels
    """

    filter_dict = {'win': 0, 'ls': 1, 'ave': 2}

    def __init__(self, name, alazar_name, filter: str = 'win',
                 numtaps: int =101, chan_b: bool = False,
                 integrate_samples: bool = False,
                 **kwargs):
        self.filter_settings = {'filter': self.filter_dict[filter],
                                'numtaps': numtaps}
        self.chan_b = chan_b
        self.number_of_channels = 2
        super().__init__(name, alazar_name, **kwargs)

        self.add_parameter(name='acquisition',
                           integrate_samples=integrate_samples,
                           parameter_class=ExpandingAlazarArrayMultiParameter)

        self._integrate_samples = integrate_samples

        self.add_parameter(name='int_time',
                           check_and_update_fn=self._update_int_time,
                           default_fn=self._int_time_default,
                           parameter_class=AcqVariablesParam)
        self.add_parameter(name='int_delay',
                           check_and_update_fn=self._update_int_delay,
                           default_fn=self._int_delay_default,
                           parameter_class=AcqVariablesParam)
        self.add_parameter(name='num_avg',
                           check_and_update_fn=self._update_num_avg,
                           default_fn= lambda : 1,
                           parameter_class=AcqVariablesParam)
        self.add_parameter(name='allocated_buffers',
                           alternative='not controllable in this controller',
                           parameter_class=NonSettableDerivedParameter)
        self.add_parameter(name='buffers_per_acquisition',
                           alternative='not controllable in this controller',
                           parameter_class=NonSettableDerivedParameter)
        self.add_parameter(name='records_per_buffer',
                           alternative='num_avg',
                           parameter_class=NonSettableDerivedParameter)
        self.add_parameter(name='samples_per_record',
                           alternative='int_time and int_delay',
                           parameter_class=NonSettableDerivedParameter)

        self.add_parameter(name='demod_freqs',
                           shape=(),
                           parameter_class=DemodFreqParameter)

        self.samples_divisor = self._get_alazar().samples_divisor


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
            sample_rate attr of acq controller to be that of alazar
            samples_per_record of acq controller
            acquisition_kwarg['samples_per_record'] of acquisition param
        """
        if (value is None) or not (0 <= value <= 0.1):
            raise ValueError('int_time must be 0 <= value <= 1')

        alazar = self._get_alazar()
        sample_rate = alazar.get_sample_rate()
        max_demod_freq = self.demod_freqs.get_max_demod_freq()
        if max_demod_freq is not None:
            self._verify_demod_freq(max_demod_freq)
        if self.int_delay() is None:
            self.int_delay.to_default()
        int_delay = self.int_delay.get()
        self._update_samples_per_record(sample_rate, value, int_delay)

    def _update_samples_per_record(self, sample_rate, int_time, int_delay):
        # update acquisition kwargs and acq controller value
        total_time = (int_time or 0) + (int_delay or 0)
        samples_needed = total_time * sample_rate
        samples_per_record = helpers.roundup(
            samples_needed, self.samples_divisor)
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
            sample_rate attr of acq controller to be that of Alazar
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
        sample_rate = alazar.get_sample_rate()
        samples_delay_min = (self.filter_settings['numtaps'] - 1)
        int_delay_min = samples_delay_min / sample_rate
        if value < int_delay_min:
            logging.warning(
                'delay is less than recommended for filter choice: '
                '(expect delay >= {})'.format(int_delay_min))

        int_time = self.int_time.get()
        self._update_samples_per_record(sample_rate, int_time, value)

    def _update_num_avg(self, value: int, **kwargs):
        if not isinstance(value, int) or value < 1:
            raise ValueError('number of averages must be a positive integer')

        self.records_per_buffer._save_val(value)
        self.buffers_per_acquisition._save_val(1)
        self.allocated_buffers._save_val(1)


    def _int_delay_default(self):
        """
        Function to generate default int_delay value

        Returns:
            minimum int_delay recommended for (numtaps - 1)
            samples to be discarded as recommended for filter
        """
        alazar = self._get_alazar()
        sample_rate = alazar.get_sample_rate()
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
        sample_rate = alazar.get_sample_rate()
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
        self.acquisition.set_setpoints_and_labels()

    def pre_start_capture(self):
        """
        Called before capture start to update Acquisition Controller with
        Alazar acquisition params and set up software wave for demodulation.
        """
        alazar = self._get_alazar()
        acq_s_p_r = self.samples_per_record.get()
        inst_s_p_r = alazar.samples_per_record.get()
        sample_rate = alazar.get_sample_rate()
        if acq_s_p_r != inst_s_p_r:
            raise Exception('acq controller samples per record {} does not match'
                            ' instrument value {}, most likely need '
                            'to set and check int_time and int_delay'.format(acq_s_p_r, inst_s_p_r))
        # if acq_s_r != inst_s_r:
        #     raise Exception('acq controller sample rate {} does not match '
        #                     'instrument value {}, most likely need '
        #                     'to set and check int_time and int_delay'.format(acq_s_r, inst_s_r))
        samples_per_record = inst_s_p_r
        demod_freqs = self.demod_freqs.get()
        # if len(demod_freqs) == 0:
        #     raise Exception('no demod_freqs set')

        records_per_buffer = alazar.records_per_buffer.get()
        self.board_info = alazar.get_idn()
        self.buffer = np.zeros(samples_per_record *
                               records_per_buffer *
                               self.number_of_channels)

        if len(demod_freqs):
            integer_list = np.arange(samples_per_record)
            angle_mat = 2 * np.pi * \
                np.outer(demod_freqs, integer_list) / sample_rate
            self.cos_mat = np.cos(angle_mat)
            self.sin_mat = np.sin(angle_mat)

    def pre_acquire(self):
        pass

    def handle_buffer(self, data):
        """
        Adds data from Alazar to buffer (effectively averaging)
        """
        self.buffer += data

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
        reshaped_buf = self.buffer.reshape(records_per_buffer,
                                           samples_per_record,
                                           self.number_of_channels)
        recordA = np.uint16(np.mean(reshaped_buf[:, :, 0], axis=0) /
                            buffers_per_acquisition)

        recordA = self._to_volts(recordA)
        unpacked = []
        if self._integrate_samples:
            unpacked.append(np.mean(recordA, axis=-1))
        else:
            unpacked.append(recordA)
        # do demodulation
        if self.demod_freqs.get_num_demods():
            magA, phaseA = self._fit(recordA)
            if self._integrate_samples:
                magA = np.mean(magA, axis=-1)
                phaseA = np.mean(phaseA, axis=-1)
            for i in range(magA.shape[0]):
                unpacked.append(magA[i])
                unpacked.append(phaseA[i])
        # same for chan b
        if self.chan_b:
            raise NotImplementedError('chan b code not complete')


        return tuple(unpacked)

    def _to_volts(self, record):
        # convert rec to volts
        bps = self.board_info['bits_per_sample']
        if bps == 12:
            volt_rec = helpers.sample_to_volt_u12(record, bps)
        else:
            logging.warning('sample to volt conversion does not exist for'
                            ' bps != 12, centered raw samples returned')
            volt_rec = record - np.mean(record)
        return volt_rec

    def _fit(self, volt_rec):
        """
        Applies low bandpass filter and demodulation fit,
        and integration limits to samples array

        Args:
            record (numpy array): record from alazar to be multiplied
                                  with the software signal, filtered and limited
                                  to ifantegration limits shape = (samples_taken, )

        Returns:
            magnitude (numpy array): shape = (demod_length, samples_after_limiting)
            phase (numpy array): shape = (demod_length, samples_after_limiting)
        """

        # volt_rec to matrix and multiply with demodulation signal matrices
        alazar = self._get_alazar()
        sample_rate = alazar.get_sample_rate()
        demod_length = self.demod_freqs.get_num_demods()
        volt_rec_mat = np.outer(np.ones(demod_length), volt_rec)
        re_mat = np.multiply(volt_rec_mat, self.cos_mat)
        im_mat = np.multiply(volt_rec_mat, self.sin_mat)

        # filter out higher freq component
        cutoff = self.demod_freqs.get_max_demod_freq() / 10
        if self.filter_settings['filter'] == 0:
            re_filtered = helpers.filter_win(re_mat, cutoff,
                                             sample_rate,
                                             self.filter_settings['numtaps'],
                                             axis=-1)
            im_filtered = helpers.filter_win(im_mat, cutoff,
                                             sample_rate,
                                             self.filter_settings['numtaps'],
                                             axis=-1)
        elif self.filter_settings['filter'] == 1:
            re_filtered = helpers.filter_ls(re_mat, cutoff,
                                            sample_rate,
                                            self.filter_settings['numtaps'],
                                            axis=-1)
            im_filtered = helpers.filter_ls(im_mat, cutoff,
                                            sample_rate,
                                            self.filter_settings['numtaps'],
                                            axis=-1)
        elif self.filter_settings['filter'] == 2:
            re_filtered = re_mat
            im_filtered = im_mat

        if self._integrate_samples:
            # apply integration limits
            beginning = int(self.int_delay() * sample_rate)
            end = beginning + int(self.int_time() * sample_rate)

            re_limited = re_filtered[:, beginning:end]
            im_limited = im_filtered[:, beginning:end]
        else:
            re_limited = re_filtered
            im_limited = im_filtered

        # convert to magnitude and phase
        complex_mat = re_limited + im_limited * 1j
        magnitude = abs(complex_mat)
        phase = np.angle(complex_mat, deg=True)

        return magnitude, phase

