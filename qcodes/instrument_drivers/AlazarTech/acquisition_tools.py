import numpy as np
from scipy import signal


def sample_to_volt_u12(self, raw_samples, input_range_volts, bps):
    # right_shift 16-bit sample by 4 to get 12 bit sample
    shifted_samples = np.right_shift(raw_samples, 4)

    # Alazar calibration
    bps = 12
    input_range_volts = 0.8
    code_zero = (1 << (bps - 1)) - 0.5
    code_range = (1 << (bps - 1)) - 0.5

    # Convert to volts
    volt_samples = (input_range_volts *
                    (shifted_samples - code_zero) / code_range)

    return volt_samples


def filter_win(rec, cutoff, sample_rate, numtaps):
    nyq_rate = sample_rate / 2.
    fir_coef = signal.firwin(numtaps, cutoff / nyq_rate)
    filtered_rec = signal.lfilter(fir_coef, 1.0, rec)
    return filtered_rec


def filter_ham(rec, cutoff, sample_rate, numtaps):
    raise NotImplementedError
    # sample_rate = self.sample_rate
    # nyq_rate = sample_rate / 2.
    # fir_coef = signal.firwin(numtaps,
    #                          cutoff / nyq_rate,
    #                          window="hamming")
    # filtered_rec = 2 * signal.lfilter(fir_coef, 1.0, rec)
    # return filtered_rec


def filter_ls(rec, cutoff, sample_rate, numtaps):
    raise NotImplementedError
    # sample_rate = self.sample_rate
    # nyq_rate = sample_rate / 2.
    # bands = [0, cutoff / nyq_rate, cutoff / nyq_rate, 1]
    # desired = [1, 1, 0, 0]
    # fir_coef = signal.firls(numtaps,
    #                         bands,
    #                         desired,
    #                         nyq=nyq_rate)
    # filtered_rec = 2 * signal.lfilter(fir_coef, 1.0, rec)
    # return filtered_rec
