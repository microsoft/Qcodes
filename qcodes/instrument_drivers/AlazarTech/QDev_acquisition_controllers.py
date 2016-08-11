from .ATS import AcquisitionController
from scipy import signal
import numpy as np


class Filtering_Controller():
    """Filtering Measurement Controller
    If 1 channel:
        Mixes reference software signal with input hardware signal,
        filters out double frequency component and returns magnitude and
        phase data for dc component. Throws away channel 2
    If 2 channels:
        Uses two hardware channels to find magnitude and
        phase data for dc component

    TODO(nataliejpg) test diffrent filters and window types
    TODO(nataliejpg) numtaps logic
    TODO(nataliejpg) cutoff logic
    TODO(nataliejpg) post aqcuire one input stuff??
    """

    def __init__(self, dif_freq, numtaps, cutoff, number_of_channels):
        self.dif_freq = dif_freq
        self.sample_rate = 50000000
        self.samples_per_record = 10024
        self.buffers_per_acquisition = None
        self.number_of_channels = number_of_channels
        self.numtaps = numtaps
        self.cutoff = cutoff
        self.delay = numtaps - 1
        self.buffer = None

    def pre_start_capture(self):
        record_duration = self.samples_per_record / self.sample_rate
        time_period_dif = 1 / self.dif_freq
        cycles_measured = record_duration / time_period_dif
        oversampling_rate = self.sample_rate / (2 * self.dif_freq)
        print("Oscillations per record: {:.2f} (expect 100+)"
              .format(cycles_measured))
        print("Oversampling rate: {:.2f} (expect > 2)"
              .format(oversampling_rate))
        print("Filtering delay is {} samples".format(self.delay))

        integer_list = np.arange(self.samples_per_record)
        if self.number_of_channels == 2:
            angle_list = (2 * np.pi * self.freq_dif / self.sample_rate *
                          integer_list)
            self.cos_list = np.cos(angle_list)
            self.sin_list = np.sin(angle_list)
        elif self.number_of_channels == 1:
            angle_shift = 90
            self.sample_shift = (2 * np.pi * self.freq_dif / self.sample_rate *
                                 angle_shift)

    def pre_acquire(self, alazar):
        # gets called after 'AlazarStartCapture'
        pass

    def handle_buffer(self, alazar, data):
        self.buffer += data

    def post_acquire(self, alazar):
        """Average over records in buffer, mix and filter double freq
        if 1 chan selected recordA is mixed with software signal
        if 2 chan selected recordA and recordB are mixed
        """
        records_per_acquisition = (1. * self.buffers_per_acquisition *
                                   self.records_per_buffer)
        recordA = np.zeros(self.samples_per_record)
        for i in range(self.records_per_buffer):
            i0 = (i * self.samples_per_record * self.number_of_channels)
            i1 = (i0 + self.samples_per_record * self.number_of_channels)
            recordA += (self.buffer[i0:i1:self.number_of_channels] /
                        records_per_acquisition)
        recordB = np.zeros(self.samples_per_record)
        for i in range(self.records_per_buffer):
            i0 = (i * self.samples_per_record * self.number_of_channels + 1)
            i1 = (i0 + self.samples_per_record * self.number_of_channels)
            recordB += (self.buffer[i0:i1:self.number_of_channels] /
                        records_per_acquisition)

        if self.number_of_channels == 1:
            mixA_re = np.multiply(self.cos_list, recordA)
            mixA_im = np.multiply(self.sin_list, recordA)
            mixA_re_filtered = filter(mixA_re)
            mixA_im_filtered = filter(mixA_im)
            mixA_complex = mixA_re_filtered + mixA_im_filtered * 1j
            magA = 2 * abs(mixA_complex)
            phaseA = 2 * np.angle(mixA_complex, deg=True)
            return magA, phaseA
        elif self.number_of_channels == 2:
            recordB_shifted = np.roll(recordB, self.sample_shift)
            mixAB_re = np.multiply(recordA, recordB)
            mixAB_im = np.multiply(recordA, recordB_shifted)
            mixAB_re_filtered = filter(mixAB_re)
            mixAB_im_filtered = filter(mixAB_im)
            mixAB_complex = mixAB_re_filtered + mixAB_im_filtered * 1j
            magAB = 2 * abs(mixAB_complex)
            phaseAB = 2 * np.angle(mixAB_complex, deg=True)
            return magAB, phaseAB

    def filter(self, rec):
        """FIR window filter applied to filter out high freq components
        """
        sample_rate = self.sample_rate
        nyq_rate = sample_rate / 2.
        fir_coef = signal.firwin(self.numtaps, self.cutoff / nyq_rate)
        filtered_rec = signal.lfilter(fir_coef, 1.0, rec)
        return filtered_rec


class HD_Controller(AcquisitionController):
    """Averaging Heterodyne Measurement Controller
    Does averaged DFT on 2 channel Alazar measurement

    TODO(nataliejpg) handling of channel number
    TODO(nataliejpg) test angle data
    """

    def __init__(self, freq_dif):
        self.freq_dif = freq_dif
        self.samples_per_record = None
        self.records_per_buffer = None
        self.buffers_per_acquisition = None
        self.number_of_channels = 2
        self.buffer = None

    def pre_start_capture(self, alazar):
        """Get config data from alazar card and set up DFT"""
        self.samples_per_record = alazar.samples_per_record()
        self.records_per_buffer = alazar.records_per_buffer()
        self.buffers_per_acquisition = alazar.buffers_per_acquisition()
        self.sample_rate = alazar.get_sample_speed()
        self.buffer = np.zeros(self.samples_per_record *
                               self.records_per_buffer *
                               self.number_of_channels)
        # TODO(nataliejpg) leave explicit or save lines? add error/logging?
        averaging = self.buffers_per_acquisition * self.records_per_buffer
        record_duration = self.samples_per_record / self.sample_rate
        time_period_dif = 1 / self.freq_dif
        cycles_measured = record_duration / time_period_dif
        oversampling_rate = self.sample_rate / (2 * self.freq_dif)
        print("Average over {:.2f} records".format(averaging))
        print("Oscillations per record: {:.2f} (expect 100+)"
              .format(cycles_measured))
        print("Oversampling rate: {:.2f} (expect > 2)"
              .format(oversampling_rate))

        integer_list = np.arange(self.samples_per_record)
        angle_list = (2 * np.pi * self.freq_dif / self.sample_rate *
                      integer_list)
        self.cos_list = np.cos(angle_list)
        self.sin_list = np.sin(angle_list)

    def pre_acquire(self, alazar):
        # gets called after 'AlazarStartCapture'
        pass

    def handle_buffer(self, alazar, data):
        self.buffer += data

    def post_acquire(self, alazar):
        """Average over records in buffer and do DFT:
        assumes samples are arranged in the buffer as follows:
        S0A, S0B, ..., S1A, S1B, ...
        with SXY the sample number X of channel Y.
        """
        records_per_acquisition = (1. * self.buffers_per_acquisition *
                                   self.records_per_buffer)
        recordA = np.zeros(self.samples_per_record)
        for i in range(self.records_per_buffer):
            i0 = (i * self.samples_per_record * self.number_of_channels)
            i1 = (i0 + self.samples_per_record * self.number_of_channels)
            recordA += (self.buffer[i0:i1:self.number_of_channels] /
                        records_per_acquisition)
        recordB = np.zeros(self.samples_per_record)
        for i in range(self.records_per_buffer):
            i0 = (i * self.samples_per_record * self.number_of_channels + 1)
            i1 = (i0 + self.samples_per_record * self.number_of_channels)
            recordB += (self.buffer[i0:i1:self.number_of_channels] /
                        records_per_acquisition)

        resA = self.fit(recordA)
        resB = self.fit(recordB)

        return resA, resB

    def fit(self, rec):
        """Do Discrete Fourier Transform and return magnitude and phase data"""
        RePart = np.dot(rec, self.cos_list) / self.samples_per_record
        ImPart = np.dot(rec, self.sin_list) / self.samples_per_record
        complex_num = RePart - ImPart * 1j
        # factor of 2 in ampl is due to loss of averaged double frequency term
        ampl = 2 * abs(complex_num)
        phase = np.angle(complex_num, deg=True)
        return [ampl, phase]


class Basic_AcquisitionController(AcquisitionController):
    """Basic AcquisitionController tested on ATS9360
    returns unprocessed data averaged by record with 2 channels
    """

    def __init__(self):
        self.samples_per_record = None
        self.records_per_buffer = None
        self.buffers_per_acquisition = None
        self.number_of_channels = 2
        self.buffer = None

    def pre_start_capture(self, alazar):
        self.samples_per_record = alazar.samples_per_record()
        self.records_per_buffer = alazar.records_per_buffer()
        self.buffers_per_acquisition = alazar.buffers_per_acquisition()
        self.buffer = np.zeros(self.samples_per_record *
                               self.records_per_buffer *
                               self.number_of_channels)

    def pre_acquire(self, alazar):
        # gets called after 'AlazarStartCapture'
        pass

    def handle_buffer(self, alazar, data):
        self.buffer += data

    def post_acquire(self, alazar):
        # average over records in buffer:
        # for ATS9360 samples are arranged in the buffer as follows:
        # S0A, S0B, ..., S1A, S1B, ...
        # with SXY the sample number X of channel Y.
        records_per_acquisition = (self.buffers_per_acquisition *
                                   self.records_per_buffer)
        recordA = np.zeros(self.samples_per_record)
        for i in range(self.records_per_buffer):
            i0 = (i * self.samples_per_record * self.number_of_channels)
            i1 = (i0 + self.samples_per_record * self.number_of_channels)
            recordA += (self.buffer[i0:i1:self.number_of_channels] /
                        records_per_acquisition)

        recordB = np.zeros(self.samples_per_record)
        for i in range(self.records_per_buffer):
            i0 = (i * self.samples_per_record * self.number_of_channels + 1)
            i1 = (i0 + self.samples_per_record * self.number_of_channels)
            recordB += (self.buffer[i0:i1:self.number_of_channels] /
                        records_per_acquisition)
        return recordA, recordB
