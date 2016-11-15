import logging
from .ATS import AcquisitionController
import numpy as np
import qcodes.utils.helpers as helpers


class HD_Controller(AcquisitionController):
    """
    This is the Acquisition Controller class which works with the ATS9360,
    averaging over buffers and records, demodulating with a software
    reference signal, averaging over the  samples.
    args:
    name: name for this acquisition_conroller as an instrument
    alazar_name: the name of the alazar instrument such that this controller
        can communicate with the Alazar
    demod_freq: the frequency of the software wave to be created
    samp_rate: the rate of sampling
    filt: the filter to be used to filter out double freq component
    chan_b: whether there is also a second channel of data to be processed
        and returned
    **kwargs: kwargs are forwarded to the Instrument base class

    TODO(nataliejpg) fix sample rate problem
    TODO(nataliejpg) test filter options
    TODO(nataliejpg) finish implementation of channel b option
    TODO(nataliejpg) make demodulation freq a param
    """

    def __init__(self, name, alazar_name, demod_freq, samp_rate=5e8,
                 filt='win', numtaps=101, chan_b=False, **kwargs):
        filter_dict = {'win': 0, 'ls': 1, 'dot': 2}
        self.demodulation_frequency = demod_freq
        self.sample_rate = samp_rate
        self.filter = filter_dict[filt]
        self.numtaps = numtaps
        self.acquisitionkwargs = {}
        self.chan_b = chan_b
        self.samples_per_record = None
        self.records_per_buffer = None
        self.buffers_per_acquisition = None
        self.number_of_channels = 2
        self.samples_delay = None
        self.samples_time = None
        self.cos_list = None
        self.sin_list = None
        self.buffer = None
        self.board_info = None
        # make a call to the parent class and by extension,
        # create the parameter structure of this class
        super().__init__(name, alazar_name, **kwargs)

        self.add_parameter(name='acquisition',
                           names=('magnitude', 'phase'),
                           units=('', ''),
                           get_cmd=self.do_acquisition)

    def update_acquisitionkwargs(self, **kwargs):
        """
        Updates the kwargs to be used when
        alazar_driver.acquire is called.

        It is also used to set the limits on the selection of
        samples used (bounded by delay and integration time)

        :param kwargs:
        :return:
        """
        samples_per_record = kwargs['samples_per_record']
        sample_rate = self.sample_rate

        if 'int_delay' in kwargs:
            int_delay = kwargs.pop('int_delay')
            samp_delay = int_delay * sample_rate
            samples_delay_min = (self.numtaps - 1)
            if samp_delay < samples_delay_min:
                int_delay_min = samples_delay_min / sample_rate
                logging.warning(
                    'delay is less than recommended for filter choice: '
                    '(expect delay >= {}'.format(int_delay_min))
        else:
            samp_delay = self.numtaps - 1

        if 'int_time' in kwargs:
            int_time = kwargs.pop('int_time')
            samp_time = int_time * sample_rate
            samples_time_max = (samples_per_record - samp_delay)
            oscilations_measured = int_time * self.demodulation_frequency
            oversampling = sample_rate / (2 * self.demodulation_frequency)
            if samp_time > samples_time_max:
                int_time_max = samples_time_max / sample_rate
                raise ValueError('int_time {} is longer than total_time - '
                                 'delay: {}'.format(int_time, int_time_max))
            elif oscilations_measured < 10:
                logging.warning('{} oscilations measured, recommend at '
                                'least 10: decrease sampling rate, take '
                                'more samples or increase demodulation '
                                'freq'.format(oscilations_measured))
            elif oversampling < 1:
                logging.warning('oversampling rate is {}, recommend > 1: '
                                'increase sampling rate or decrease '
                                'demodulation frequency'.format(oversampling))
        else:
            samp_time = samples_per_record - samp_delay

        self.samples_time = int(samp_time)
        self.samples_delay = int(samp_delay)

        self.acquisitionkwargs.update(**kwargs)

    def do_acquisition(self):
        """
        this method performs an acquisition, which is the get_cmd for the
        acquisiion parameter of this instrument
        :return:
        """
        values = self._get_alazar().acquire(acquisition_controller=self,
                                            **self.acquisitionkwargs)
        return values

    def pre_start_capture(self):
        """
        Called before capture start to update Acquisition Controller with
        alazar acquisition params and set up software wave for demodulation.
        :return:
        """
        alazar = self._get_alazar()
        self.samples_per_record = alazar.samples_per_record.get()
        self.records_per_buffer = alazar.records_per_buffer.get()
        self.buffers_per_acquisition = alazar.buffers_per_acquisition.get()
        self.board_info = alazar.get_idn()
        self.buffer = np.zeros(self.samples_per_record *
                               self.records_per_buffer *
                               self.number_of_channels,
                               dtype=np.uint16)

        integer_list = np.arange(self.samples_per_record, dtype=np.uint16)
        angle_list = (2 * np.pi * self.demodulation_frequency /
                      self.sample_rate * integer_list)

        self.cos_list = np.cos(angle_list)
        self.sin_list = np.sin(angle_list)

    def pre_acquire(self):
        """
        See AcquisitionController
        :return:
        """
        pass

    def handle_buffer(self, data):
        """
        Adds data from alazar to buffer (effectively averaging)
        :return:
        """
        self.buffer += data

    def post_acquire(self):
        """
        Processes the data according to ATS9360 settings, splitting into
        records and averaging over them, then applying demodulation fit
        nb: currently only channel A
        :return: samples_magnitude_array, samples_phase_array
        """
        records_per_acquisition = (self.buffers_per_acquisition *
                                   self.records_per_buffer)
        # for ATS9360 samples are arranged in the buffer as follows:
        # S00A, S00B, S01A, S01B...S10A, S10B, S11A, S11B...
        # where SXYZ is record X, sample Y, channel Z.

        # break buffer up into records and averages over them
        recA = np.zeros(self.samples_per_record)
        for i in range(self.records_per_buffer):
            i0 = (i * self.samples_per_record * self.number_of_channels)
            i1 = (i0 + self.samples_per_record * self.number_of_channels)
            recA += self.buffer[i0:i1:self.number_of_channels]
        recordA = np.uint16(recA / records_per_acquisition)

        # do demodulation
        magA, phaseA = self.fit(recordA)

        # same for chan b
        if self.chan_b:
            raise NotImplementedError('chan b code not complete')
            # recordB = np.zeros(self.samples_per_record, dtype=np.uint16)
            # for i in range(self.records_per_buffer):
            #     i0 = (i * self.samples_per_record *
            #           self.number_of_channels + 1)
            #     i1 = (i0 + self.samples_per_record * self.number_of_channels)
            #     recordB += np.uint16(self.buffer[i0:i1:self.number_of_channels] /
            #                          records_per_acquisition)
            # magB, phaseB = self.fit(recordB)

        return magA, phaseA

    def fit(self, rec):
        """
        Applies volts conversion, demodulation fit, low bandpass filter
        and integration limits to samples array and averages
        :return: magnitude, phase
        """

        # convert rec to volts
        bps = self.board_info['bits_per_sample']
        if bps == 12:
            volt_rec = sample_to_volt_u12(rec, bps)
        else:
            logging.warning('sample to volt conversion does not exist for'
                            ' bps != 12, centered raw samples returned')
            volt_rec = rec - np.mean(rec)

        # multiply with software wave
        re_wave = np.multiply(volt_rec, self.cos_list)
        im_wave = np.multiply(volt_rec, self.sin_list)
        cutoff = self.demodulation_frequency / 10

        # filter out higher freq component
        if self.filter == 0:
            re_filtered = helpers.filter_win(re_wave, cutoff,
                                             self.sample_rate, self.numtaps)
            im_filtered = helpers.filter_win(im_wave, cutoff,
                                             self.sample_rate, self.numtaps)
        elif self.filter == 1:
            re_filtered = helpers.filter_ls(re_wave, cutoff,
                                            self.sample_rate, self.numtaps)
            im_filtered = helpers.filter_ls(im_wave, cutoff,
                                            self.sample_rate, self.numtaps)
        elif self.filter == 2:
            re_filtered = re_wave
            im_filtered = im_wave

        # apply integration limits
        start = self.samples_delay
        end = start + self.samples_time

        re_limited = re_filtered[start:end]
        im_limited = im_filtered[start:end]

        # convert to magnitude and phase
        complex_num = re_limited + im_limited * 1j
        mag = np.mean(abs(complex_num))
        phase = np.mean(np.angle(complex_num, deg=True))

        return mag, phase


def sample_to_volt_u12(raw_samples, bps):
    """
    Applies volts conversion for 12 bit sample data stored
    in 2 bytes
    :return: samples_magnitude_array, samples_phase_array
    """

    # right_shift 16-bit sample by 4 to get 12 bit sample
    shifted_samples = np.right_shift(raw_samples, 4)

    # Alazar calibration
    code_zero = (1 << (bps - 1)) - 0.5
    code_range = (1 << (bps - 1)) - 0.5

    # TODO(nataliejpg) make this not hard coded
    input_range_volts = 1
    # Convert to volts
    volt_samples = np.float64(input_range_volts *
                              (shifted_samples - code_zero) / code_range)

    return volt_samples
