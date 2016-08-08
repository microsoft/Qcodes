from .ATS import AcquisitionController
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


class Controller():
    def __init__(self, dif_freq):
        self.dif_freq = dif_freq
        self.sample_rate = 5000000
        self.samples_per_record = 10024
        
        record_duration = self.samples_per_record/self.sample_rate
        dif_time_period = 1/self.dif_freq
        cycles_measured = record_duration / dif_time_period
        oversampling_rate = self.sample_rate/(2*self.dif_freq)
        print("Measuring from "+str(cycles_measured)+" samples should be on the order of 100 at least")
        print("Oversampling rate is "+str(oversampling_rate)+" should be > 2")

    def pre_start_capture(self):
        self.integer_list = np.arange(self.samples_per_record)
        self.angle_list = (2 * np.pi * self.dif_freq / self.sample_rate *
                      self.integer_list)

        self.cos_list = np.cos(self.angle_list)
        self.sin_list = np.sin(self.angle_list)
                              
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
        records_per_acquisition = (1. * self.buffers_per_acquisition *
                                   self.records_per_buffer)
        recordA = np.zeros(self.samples_per_record)
        for i in range(self.records_per_buffer):
            i0 = i * self.samples_per_record * self.number_of_channels
            i1 = i0 + self.samples_per_record * self.number_of_channels
            recordA += self.buffer[i0:i1:self.number_of_channels] / records_per_acquisition

        recordB = np.zeros(self.samples_per_record)
        for i in range(self.records_per_buffer):
            i0 = (i * self.samples_per_record * self.number_of_channels) + 1
            i1 = i0 + self.samples_per_record *self.number_of_channels
            recordB += self.buffer[i0:i1:self.number_of_channels] / records_per_acquisition
        
        resA = self.fit(recordA)
        resB = self.fit(recordB)
        
        return resA, resB

def make_rec(controller, freq, phase_deg):
    phase_rad = phase_deg*2*np.pi/180
    angle_list = (2 * np.pi * freq / controller.sample_rate *
                  controller.integer_list)+phase_rad
    rec = np.cos(angle_list)
    return rec

def main(freq1, freq2):
    c = Controller(freq1)
    c.pre_start_capture()
    rec1 = make_rec(c, freq2, 0)
    rec2 = c.cos_list
    plt.figure(1)
    plt.subplot(211)
    plt.plot(rec1)
    plt.subplot(212)
    plt.plot(rec1)
    plt.show()
    s = np.multiply(rec1, rec2)
    plt.plot(s)
    plt.show()


n = 61
a = signal.firwin(n, cutoff = 0.3, window = "hamming")
#Frequency and phase response
mfreqz(a)
show()
#Impulse and step response
figure(2)
impz(a)
show()

    # import qcodes.instrument_drivers.AlazarTech.local_test2 as cont
