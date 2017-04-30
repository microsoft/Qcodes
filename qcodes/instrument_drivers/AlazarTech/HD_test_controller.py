from .ATS import AcquisitionController
import math
import numpy as np
import matplotlib.pyplot as plt

# HD_ test controller
class Controller():
    def __init__(self, dif_freq):
        self.dif_freq = dif_freq
        self.sample_rate = 10000000
        self.samples_per_record = 2024
        
        record_duration = self.samples_per_record / self.sample_rate
        time_period_dif = 1 / self.dif_freq
        cycles_measured = record_duration / time_period_dif
        oversampling_rate = self.sample_rate / (2 * self.dif_freq)
        print("Oscillations per record: {:.2f} (expect 100+)".format(cycles_measured))
        print("Oversampling rate: {:.2f} (expect > 2)".format(oversampling_rate))

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
        complex_num = RePart - ImPart*1j
        ampl = 2 * abs(complex_num)
        phase = np.angle(complex_num, deg=True)
        
        return [ampl, phase]

def make_rec(controller, freq, phase_deg):
    phase_rad = phase_deg*np.pi/180
    angle_list = (2 * np.pi * freq / controller.sample_rate *
                  controller.integer_list)+phase_rad
    rec = np.cos(angle_list)
    return rec

def main(freq, phase_dif):
    # make an acqisition controller and set up sin and cos software signals
    c = Controller(freq)
    c.pre_start_capture()

    # make a 'hardware' signal with same frequency shifted by phase
    # mix with software signals to make real and imaginary parts of mixed signal
    rec = make_rec(c, freq, phase_dif)
    cos_rec = np.multiply(c.cos_list, rec)
    sin_rec = np.multiply(c.sin_list, rec)

    # print averaged magnitude and phase of mixed signal
    # should give dc part of signal as double frequency part averages out
    print(" mag, phase : "+str(c.fit(rec)))
    plt.figure(1)
    plt.subplot(311)
    plt.plot(c.cos_list[:100], "-r")
    plt.plot(c.sin_list[:100], "-g")
    plt.title('"software" signals')
    plt.subplot(312)
    plt.plot(rec[:100])
    plt.title('"hardware" signal')
    plt.subplot(313)
    plt.plot(cos_rec[:100], "-r")
    plt.plot(sin_rec[:100], "-g")
    plt.title('re and im parts of mixed signal')
    plt.show()

    # sweep 'hardware' signal frequency through software signal and 
    # observe magnitude and phase response
    point_ran = 10000
    sweep_mag = [[],[]]
    for i in range(point_ran+1):
        freq_ran = 0.2*freq
        shift_freq = (freq_ran/point_ran)*(-point_ran/2+i)
        rec = make_rec(c, freq+shift_freq, 0)
        mag, phase = c.fit(rec)
        sweep_mag[1].append(mag)
        sweep_mag[0].append(shift_freq)
    sweep_phase = [[],[]]
    for i in range(point_ran+1):
        angle_ran = 45
        shift_phase = (angle_ran/point_ran)*(-point_ran/2+i)
        rec = make_rec(c, freq, phase_dif+shift_phase)
        mag, phase = c.fit(rec)
        sweep_phase[1].append(phase)
        sweep_phase[0].append(shift_phase)
    plt.figure(1)
    plt.subplot(211)
    plt.title('sweep:magnitude response')
    plt.plot(sweep_mag[0], sweep_mag[1])
    plt.subplot(212)
    plt.title('sweep:phase response')
    plt.plot(sweep_phase[0], sweep_phase[1])
    plt.show()



    # import qcodes.instrument_drivers.AlazarTech.HD_test_controller as cont
