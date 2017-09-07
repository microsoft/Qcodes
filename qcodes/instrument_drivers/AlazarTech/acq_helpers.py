import numpy as np
import math

def sample_to_volt_u12(raw_samples, bps, input_range_volts):
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

    # Convert to volts
    volt_samples = np.float64(input_range_volts *
                              (shifted_samples - code_zero) / code_range)

    return volt_samples


def roundup(num, to_nearest):
    """
    Rounds up the 'num' to the nearest multiple of 'to_nearest', all int

    Args:
        num: to be rounded up
        to_nearest: value to be rounded to int multiple of
    return:
        rounded up value

    """
    # remainder = num % to_nearest
    # divisor = num // to_nearest
    # return int(num if remainder == 0 else (divisor + 1)*to_nearest)
    # there seems to be alignment related issues with non power of two
    # buffers so restrict to power of two for now
    smallest_power = 2 ** math.ceil(math.log2(num))
    return max(smallest_power, 256)