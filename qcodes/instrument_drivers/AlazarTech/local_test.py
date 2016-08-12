from .ATS import AcquisitionController
import math
import numpy as np
import matplotlib.pyplot as plt


class Controller():
    def __init__(self, dif_freq):
        self.dif_freq = dif_freq
        self.sample_rate = 500000000
        self.samples_per_record = 1024
        
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

    def fit(self, rec):
        # Discrete Fourier Transform
        RePart = np.dot(rec, self.cos_list) / self.samples_per_record
        ImPart = np.dot(rec, self.sin_list) / self.samples_per_record
        # factor of 2 is because amplitude is split between finite term and 
        # double frequency term which averages to 0
        ampl = 2 * np.sqrt(RePart ** 2 + ImPart ** 2)
        phase = math.atan2(ImPart, RePart) * 180 / (2 * math.pi)
        
        return [ampl, phase]
                              
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

def main(freq1, phase):
    c = Controller(freq1)
    c.pre_start_capture()
    ran = 100
    l = [[],[]]
    for i in range(ran+1):
        freq = freq1-0.002*freq1*(ran/2-i)
        rec = make_rec(c, freq, phase)
        mag, phase = c.fit(rec)
        l[0].append(mag)
        l[1].append(phase)
    plt.figure(1)
    plt.subplot(211)
    plt.plot(l[0])
    plt.subplot(212)
    plt.plot(l[1])
    plt.show()
    # cos_rec = np.multiply(c.cos_list, rec)
    # sin_rec = np.multiply(c.sin_list, rec)
    #print(c.fit(rec))
    # plt.figure(1)
    # plt.subplot(511)
    # plt.plot(c.cos_list)
    # plt.subplot(512)
    # plt.plot(c.sin_list)
    # plt.subplot(513)
    # plt.plot(rec)
    # plt.subplot(514)
    # plt.plot(cos_rec)
    # plt.subplot(515)
    # plt.plot(sin_rec)
    # plt.show()
