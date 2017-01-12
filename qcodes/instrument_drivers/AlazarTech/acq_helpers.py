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
