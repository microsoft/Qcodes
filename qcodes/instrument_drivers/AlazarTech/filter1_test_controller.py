import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# one channel filter controller
class Controller():
    def __init__(self, dif_freq):
        self.dif_freq = dif_freq
        self.sample_rate = 500000000
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

def filter(controller, rec, numtaps, cutoff):
    sample_rate = controller.sample_rate
    print("sample_rate is "+str(sample_rate))
    nyq_rate = sample_rate / 2.
    print("nyq_rate is "+ str(nyq_rate))
    delay = (numtaps-1)
    fir_coef = signal.firwin(numtaps, cutoff/nyq_rate, window = "hamming")
    filtered_rec = 2 * signal.lfilter(fir_coef, 1.0, rec)
    w, h = signal.freqz(fir_coef)
    plt.plot(w*controller.sample_rate/(2*np.pi), 20 * np.log10(abs(h)), 'b')
    plt.ylabel('Amplitude [dB]', color='b')
    plt.xlabel('Frequency [rad/sample]')
    plt.show()
    return fir_coef, filtered_rec, delay

def make_rec(controller, freq, ampdif, phase_deg):
    phase_rad = phase_deg*2*np.pi/360
    angle_list = (2 * np.pi * freq * controller.integer_list / controller.sample_rate)+phase_rad
    rec = ampdif*np.cos(angle_list)
    return rec

def main(freq1, freq2, phasedif, ampdif, numtaps, cutoff):
    # make an acquisition controller (and sampling rate warnings related to 
    # given freq1) and integer list used for signal making
    c = Controller(freq1)
    c.pre_start_capture()

    # make one hardware signal  with given frequency and phase difference 
    # make a software signal with real and imaginary parts
    # plot first 100 points of each
    rec1 = make_rec(c, freq2, phase_deg=phasedif, ampdif=ampdif)
    plt.figure(1)
    plt.subplot(211)
    plt.plot(rec1[:100])
    plt.subplot(212)
    plt.plot(rec2[:100])
    plt.show()

    # mix hardware signals and filter out frequencies above cutoff
    # should leave only freq_dif component
    rec12 = np.multiply(rec1, rec2)
    fir_coef, filtered_rec_12 = filter(c, rec12, numtaps, cutoff)
    delay = numtaps-1

    # plot filter
    w, h = signal.freqz(fir_coef)
    freq_axis = (w/np.pi)*c.sample_rate
    fig = plt.figure(1)
    ax1 = fig.add_subplot(111)
    plt.title('Digital filter frequency response')
    plt.plot(freq_axis, 20 * np.log10(abs(h)), 'b')
    plt.ylabel('Amplitude [dB]', color='b')
    plt.xlabel('Frequency [Hz]')

    ax2 = ax1.twinx()
    angles = np.unwrap(np.angle(h))
    plt.plot(freq_axis, angles, 'g')
    plt.ylabel('Angle (radians)', color='g')
    plt.grid()
    plt.axis('tight')
    plt.show()


    # shift first hardware signal by 90 degrees to make
    # sin signal for mixing, mix and plot
    rec3 = make_rec(c, freq1, sin=True)
    rec23 = np.multiply(rec2, rec3)
    fir_coef, filtered_rec_23 = filter(c, rec23, numtaps, cutoff)

    # plot filtered signal and magnitude and phase results
    plt.figure(1)
    plt.subplot(411)
    plt.title('Re part')
    plt.plot(rec12, '-r')
    plt.ylabel('unfiltered record', color='r')
    plt.plot(filtered_rec_12, '-g')
    plt.ylabel('filtered record', color='g')
    plt.subplot(412)
    plt.title('Im part')
    plt.plot(rec23, '-r')
    plt.ylabel('unfiltered record', color='r')
    plt.plot(filtered_rec_23, '-g')
    plt.ylabel('filtered record', color='g')
    
    # add to plot magnitude and phase of resulting complex number
    complex_rec = filtered_rec_12+filtered_rec_23*1j
    magnitude = abs(complex_rec)
    phase = np.angle(complex_rec, deg=True)
    plt.subplot(413)
    plt.title('Magnitude')
    plt.plot(magnitude)
    plt.ylabel('magnitude')
    plt.subplot(414)
    plt.title('Phase')
    plt.plot(phase)
    plt.ylabel('phase')
    plt.show()

