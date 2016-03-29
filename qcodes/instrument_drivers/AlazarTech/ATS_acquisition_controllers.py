from .ATS import AcquisitionController
import math
import numpy as np


# DFT AcquisitionController
class DFT_AcquisitionController(AcquisitionController):
    def __init__(self, demodulation_frequency, samples_per_record, records_per_buffer, buffers_per_acquisition):
        self.demodulation_frequency = demodulation_frequency
        self.samples_per_record = samples_per_record
        self.records_per_buffer = records_per_buffer
        self.buffers_per_acquisition = buffers_per_acquisition
        # TODO (S) this is not very general:
        self.number_of_channels = 2
        self.cos_list = None
        self.sin_list = None
        self.buffer = None

    def pre_start_capture(self, alazar):
        sample_speed = alazar.get_sample_speed()
        integer_list = np.arange(self.samples_per_record)

        self.cos_list = np.cos(2 * np.pi * self.demodulation_frequency / sample_speed * integer_list)
        self.sin_list = np.sin(2 * np.pi * self.demodulation_frequency / sample_speed * integer_list)
        self.buffer = np.zeros(self.samples_per_record * self.records_per_buffer * self.number_of_channels)

    def pre_acquire(self, alazar):
        # this could be used to start an Arbitrary Waveform Generator, etc...
        # using this method ensures that the contents are executed AFTER the Alazar card starts listening for a
        # trigger pulse
        pass

    def handle_buffer(self, alazar, data):
        self.buffer += data

    def post_acquire(self, alazar):
        # average all records in a buffer
        recordA = np.zeros(self.samples_per_record)
        for i in range(self.records_per_buffer):
            recordA += self.buffer[i * self.samples_per_record: (i + 1) * self.samples_per_record] / (
                1. * self.buffers_per_acquisition * self.records_per_buffer)

        recordB = np.zeros(self.samples_per_record)
        for i in range(self.records_per_buffer):
            recordB += self.buffer[
                       i * self.samples_per_record + len(self.buffer) / 2: (i + 1) * self.samples_per_record + len(
                           self.buffer) / 2] / (1. * self.buffers_per_acquisition * self.records_per_buffer)

        if self.number_of_channels == 2:
            # fit channel A and channel B
            res1 = self.fit(recordA)
            res2 = self.fit(recordB)
            return [alazar.signal_to_volt(1, res1[0] + 127.5), alazar.signal_to_volt(2, res2[0] + 127.5), res1[1], res2[1],
                    (res1[1] - res2[1]) % 360]
        else:
            raise Exception("Could not find CHANNEL_B during data extraction")
        return None

    def fit(self, buf):
        # Discrete Fourier Transform
        RePart = np.dot(buf - 127.5, self.cos_list) / float(self.samples_per_record)
        ImPart = np.dot(buf - 127.5, self.sin_list) / float(self.samples_per_record)

        # the factor of 2 in the amplitude is due to the fact that there is a negative frequency as well
        ampl = 2 * np.sqrt(RePart ** 2 + ImPart ** 2)

        # see manual page 52!!! (using unsigned data)
        return [ampl, math.atan2(ImPart, RePart) * 360 / (2 * math.pi)]
