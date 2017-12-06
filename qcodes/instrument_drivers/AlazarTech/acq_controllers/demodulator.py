import numpy as np
from scipy import signal
import logging
logger = logging.getLogger(__name__)

def filter_win(rec, cutoff, sample_rate, numtaps, axis=-1):
    """
    low pass filter, returns filtered signal using FIR window
    filter

    Args:
        rec: record to filter
        cutoff: cutoff frequency
        sample_rate: sampling rate
        numtaps: number of frequency comppnents to use in the filer
        axis: axis of record to apply filter along
    """
    nyq_rate = sample_rate / 2.
    fir_coef = signal.firwin(numtaps, cutoff / nyq_rate)
    filtered_rec = signal.lfilter(fir_coef, [1.0], rec, axis=axis)
    return filtered_rec


def filter_ls(rec, cutoff, sample_rate, numtaps, axis=-1):
    """
    low pass filter, returns filtered signal using FIR
    least squared filter

    Args:
        rec: record to filter
        cufoff: cutoff frequency
        sample_rate: sampling rate
        numtaps: number of frequency comppnents to use in the filer
        axis: axis of record to apply filter along
    """
    raise NotImplementedError


def filter_ham(rec, cutoff, sample_rate, numtaps):
    raise NotImplementedError
    # sample_rate = self.sample_rate
    # nyq_rate = sample_rate / 2.
    # fir_coef = signal.firwin(numtaps,
    #                          cutoff / nyq_rate,
    #                          window="hamming")
    # filtered_rec = 2 * signal.lfilter(fir_coef, 1.0, rec)
    # return filtered_rec

class Demodulator:



    def __init__(self,
                 buffers_per_acquisition: int,
                 records_per_buffer: int,
                 samples_per_record: int,
                 sample_rate: float,
                 filter_settings,
                 demod_freqs,
                 average_buffers: bool=True,
                 average_records: bool=True,
                 integrate_samples: bool=True):

        self.filter_settings = filter_settings
        self.sample_rate = sample_rate
        if average_buffers:
            len_buffers = 1
        else:
            len_buffers = buffers_per_acquisition

        if average_records:
            len_records = 1
        else:
            len_records = records_per_buffer

        num_demods = len(demod_freqs)
        self.demod_freqs = np.array(demod_freqs)
        mat_shape = (num_demods, len_buffers,
                     len_records, samples_per_record)
        self.mat_shape = mat_shape
        integer_list = np.arange(samples_per_record)
        integer_mat = (np.outer(np.ones(len_buffers),
                                np.outer(np.ones(len_records), integer_list)))
        angle_mat = 2 * np.pi * \
                    np.outer(demod_freqs, integer_mat).reshape(mat_shape) / sample_rate
        self.cos_mat = np.cos(angle_mat)
        self.sin_mat = np.sin(angle_mat)
        self.integrate_samples = integrate_samples

    def demodulate(self, volt_rec, int_delay, int_time):
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
        demod_length = len(self.demod_freqs)
        volt_rec_mat = np.outer(np.ones(demod_length), volt_rec).reshape(self.mat_shape)
        re_mat = np.multiply(volt_rec_mat, self.cos_mat)
        im_mat = np.multiply(volt_rec_mat, self.sin_mat)*0

        # filter out higher freq component
        cutoff = max(self.demod_freqs)/10
        if self.filter_settings['filter'] == 0:
            re_filtered = filter_win(re_mat, cutoff,
                                     self.sample_rate,
                                     self.filter_settings['numtaps'],
                                     axis=-1)
            im_filtered = filter_win(im_mat, cutoff,
                                     self.sample_rate,
                                     self.filter_settings['numtaps'],
                                     axis=-1)
        elif self.filter_settings['filter'] == 1:
            re_filtered = filter_ls(re_mat, cutoff,
                                    self.sample_rate,
                                    self.filter_settings['numtaps'],
                                    axis=-1)
            im_filtered = filter_ls(im_mat, cutoff,
                                    self.sample_rate,
                                    self.filter_settings['numtaps'],
                                    axis=-1)
        elif self.filter_settings['filter'] == 2:
            re_filtered = re_mat
            im_filtered = im_mat
        else:
            raise RuntimeError("Filter setting: {} not implemented".format(self.filter_settings['filter']))

        if self.integrate_samples:
            # apply integration limits
            beginning = int(int_delay * self.sample_rate)
            end = beginning + int(int_time * self.sample_rate)

            re_limited = re_filtered[..., beginning:end]
            im_limited = im_filtered[..., beginning:end]
        else:
            re_limited = re_filtered
            im_limited = im_filtered

        # convert to magnitude and phase
        complex_mat = re_limited + im_limited * 1j
        magnitude = abs(complex_mat)
        phase = np.angle(complex_mat, deg=True)

        return magnitude, phase

    @staticmethod
    def verify_demod_freq(value, sample_rate, int_time):
        """
        Function to validate a demodulation frequency

        Checks:
            - 1e6 <= value <= 500e6
            - number of oscillation measured using current 'int_time' param value
              at this demodulation frequency value
            - oversampling rate for this demodulation frequency

        Args:
            value: proposed demodulation frequency
            sample_rate: Rate with witch data is sampled
            int_time: total time that data is sampled over used for demodulation
        Returns:
            bool: Returns True if suitable number of oscillations are measured and
            oversampling is > 1, False otherwise.
        Raises:
            ValueError: If value is not a valid demodulation frequency.
        """
        if (value is None) or not (1e6 <= value <= 500e6):
            raise ValueError('demod_freqs must be 1e6 <= value <= 500e6')
        isValid = True
        min_oscillations_measured = int_time * value
        oversampling = sample_rate / (2 * value)
        if min_oscillations_measured < 10:
            isValid = False
            logger.warning('{} oscillation measured for largest '
                            'demod freq, recommend at least 10: '
                            'decrease sampling rate, take '
                            'more samples or increase demodulation '
                            'freq'.format(min_oscillations_measured))
        elif oversampling < 1:
            isValid = False
            logger.warning('oversampling rate is {}, recommend > 1: '
                            'increase sampling rate or decrease '
                            'demodulation frequency'.format(oversampling))

        return isValid