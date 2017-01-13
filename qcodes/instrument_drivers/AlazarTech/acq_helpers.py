import numpy as np
from scipy import signal


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
    
def roundup(num, to_nearest):
    return num if num % to_nearest == 0 else num + to_nearest - num % to_nearest
