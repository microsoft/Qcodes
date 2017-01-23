import numpy as np
from scipy import signal
import logging


def filter_win(rec, cutoff, sample_rate, numtaps, axis=-1):
    """
    low pass filter, returns filtered signal using FIR window
    filter

    inputs:
        record to filter
        cutoff frequency
        sampling rate
        number of frequency comppnents to use in the filer
        axis of record to apply filter along
    """
    nyq_rate = sample_rate / 2.
    fir_coef = signal.firwin(numtaps, cutoff / nyq_rate)
    filtered_rec = signal.lfilter(fir_coef, 1.0, rec, axis=axis)
    return filtered_rec


def filter_ls(rec, cutoff, sample_rate, numtaps, axis=-1):
    """
    low pass filter, returns filtered signal using FIR
    least squared filter

    inputs:
        record to filter
        cutoff frequency
        sampling rate
        number of frequency comppnents to use in the filer
        axis of record to apply filter along
    """
    raise NotImplementedError


def sample_to_volt_u12(raw_samples, bps):
    """
    Applies volts conversion for 12 bit sample data stored
    in 2 bytes
    return:
        samples_magnitude_array
        samples_phase_array
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


def roundup(num, to_nearest):
    """
    Rounds up the 'num' to the nearest multiple of 'to_nearest', all int

    inputs:
        num to be rounded up
        to_nearest value to be rounded to int multiple of
    return:
        rounded up value
    """
    remainder = num % to_nearest
    return int(num if remainder == 0 else num + to_nearest - remainder)


def fit(acq_controller, record, averaging=False):
    """
    Applies volts conversion, demodulation fit, low bandpass filter
    and integration limits to samples array

    Args:
        rec (numpy array): record from alazar to be multiplied with the
                           software signal, filtered and limited to
                           integration limits
                           shape = (samples_taken, )

    Returns:
        magnitude (numpy array): shape = (demod_length, samples_used)
        phase (numpy array): shape = (demod_length, samples_used)
    """
    # convert rec to volts
    bps = acq_controller.board_info['bits_per_sample']
    if bps == 12:
        volt_rec = sample_to_volt_u12(record, bps)
    else:
        logging.warning('sample to volt conversion does not exist for'
                        ' bps != 12, centered raw samples returned')
        volt_rec = record - np.mean(record)

    # volt_rec to matrix and multiply with demodulation signal matrices
    volt_rec_mat = np.outer(np.ones(acq_controller._demod_length), volt_rec)
    re_mat = np.multiply(volt_rec_mat, acq_controller.cos_mat)
    im_mat = np.multiply(volt_rec_mat, acq_controller.sin_mat)

    # filter out higher freq component
    cutoff = acq_controller.get_max_demod_freq() / 10
    if acq_controller.filter_settings['filter'] == 0:
        re_filtered = filter_win(re_mat, cutoff,
                                 acq_controller.sample_rate,
                                 acq_controller.filter_settings[
                                     'numtaps'],
                                 axis=1)
        im_filtered = filter_win(im_mat, cutoff,
                                 acq_controller.sample_rate,
                                 acq_controller.filter_settings[
                                     'numtaps'],
                                 axis=1)
    elif acq_controller.filter_settings['filter'] == 1:
        re_filtered = filter_ls(re_mat, cutoff,
                                acq_controller.sample_rate,
                                acq_controller.filter_settings[
                                    'numtaps'],
                                axis=1)
        im_filtered = filter_ls(im_mat, cutoff,
                                acq_controller.sample_rate,
                                acq_controller.filter_settings[
                                    'numtaps'],
                                axis=1)

    # apply integration limits
    beginning = int(acq_controller.int_delay() * acq_controller.sample_rate)
    end = beginning + int(acq_controller.int_time() *
                          acq_controller.sample_rate)

    re_limited = re_filtered[:, beginning:end]
    im_limited = im_filtered[:, beginning:end]

    # convert to magnitude and phase
    complex_mat = re_limited + im_limited * 1j
    magnitude = abs(complex_mat)
    phase = np.angle(complex_mat, deg=True)

    if averaging:
        magnitude = np.mean(magnitude, axis=1)
        phase = np.mean(phase, axis=1)

    return magnitude, phase
