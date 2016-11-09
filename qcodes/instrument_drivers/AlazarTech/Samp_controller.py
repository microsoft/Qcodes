from .ATS import AcquisitionController
import numpy as np
from qcodes import Parameter
from scipy import signal

#import acquistion_tools as tools


class SamplesParam(Parameter):
    """
    Hardware controlled parameter class for Alazar acquisition. To be used with
    HD_Samples_Controller (tested with ATS9360 board) for return of an array of
    sample data from the Alazar, averaged over records and buffers.

    TODO(nataliejpg) fix setpoints/shapes horriblenesss
    """

    def __init__(self, name, instrument):
        super().__init__(name)
        self._instrument = instrument
        self.acquisitionkwargs = {}
        self.names = ('magnitude', 'phase')
        self.units = ('', '')
        self.setpoint_names = (('sample_num',), ('sample_num',))
        self.setpoints = ((1,), (1,))
        self.shapes = ((1,), (1,))

    def update_acquisition_kwargs(self, samp_time=None, **kwargs):
        # needed to update config of the software parameter on sweep change
        # freq setpoints tuple as needs to be hashable for look up
        if samp_time:
            npts = samp_time
            n = tuple(np.arange(npts))
            self.setpoints = ((n,), (n,))
            self.shapes = ((npts,), (npts,))
        elif 'samples_per_record' in kwargs:
            npts = kwargs['samples_per_record']
            n = tuple(np.arange(npts))
            self.setpoints = ((n,), (n,))
            self.shapes = ((npts,), (npts,))
        else:
            raise ValueError('samples_per_record or sample_num must be specified')
        # updates dict to be used in acquisition get call
        self.acquisitionkwargs.update(**kwargs)
    
    def get(self):
        mag, phase = self._instrument._get_alazar().acquire(
            acquisition_controller=self._instrument,
            **self.acquisitionkwargs)
        return mag, phase


class HD_Samples_Controller(AcquisitionController):
    """
    This is the Acquisition Controller class which works with the ATS9360,
    averaging over buffers and records and demodulating with a software
    reference signal, returning the  samples.
    args:
    name: name for this acquisition_conroller as an instrument
    alazar_name: the name of the alazar instrument such that this controller
        can communicate with the Alazar
    demod_freq: the frequency of the software wave to be created
    samp_rate: the rate of sampling
    filt: the filter to be used to filter out double freq component
    chan_b: whether there is also a second channel of data to be processed
        and returned
    **kwargs: kwargs are forwarded to the Instrument base class

    TODO(nataliejpg) fix sample rate problem
    TODO(nataliejpg) test filter options
    TODO(nataliejpg) test mag phase logic
    TODO(nataliejpg) finish implementation of channel b option
    TODO(nataliejpg) update docstrings :P
    """

    def __init__(self, name, alazar_name, demod_freq, samp_rate=500e6, filt='win',
                 numtaps=101, chan_b=False, **kwargs):
        filter_dict = {'win': 0, 'hamming': 1, 'ls': 2}
        self.demodulation_frequency = demod_freq
        self.sample_rate = samp_rate
        self.filter = filter_dict[filt]
        self.int_time = None
        self.numtaps = numtaps
        self.chan_b = chan_b
        self.samples_per_record = 0
        self.records_per_buffer = 0
        self.buffers_per_acquisition = 0
        self.number_of_channels = 2
        self.samples_delay = None
        self.samples_time = None
        self.cos_list = None
        self.sin_list = None
        self.buffer = None
        self.board_info = None
        # make a call to the parent class and by extension,
        # create the parameter structure of this class
        super().__init__(name, alazar_name, **kwargs)

        self.add_parameter(name='acquisition',
                           parameter_class=SamplesParam)

    def update_acquisitionkwargs(self, **kwargs):
        """
        This method must be used to update the kwargs used for the acquisition
        with the alazar_driver.acquire
        :param kwargs:
        :return:
        """
        samples_per_record = kwargs['samples_per_record']
        sample_rate = self.sample_rate
        
        if 'int_delay' in kwargs:
            int_delay = kwargs['int_delay']
            samp_delay = int_delay * sample_rate
            samples_delay_min = (self.numtaps - 1)
            if samp_delay < samples_delay_min:
                int_delay_min = samples_delay_min / sample_rate
                Warning(
                    'delay is less than recommended for filter choice: '
                    '(expect delay >= {}'.format(int_delay_min))
        else:
            samp_delay = self.numtaps - 1
        
        if 'int_time' in kwargs:
            int_time = kwargs['int_time']
            samp_time = int_time * sample_rate
            samples_time_max = (samples_per_record - samp_delay)
            oscilations_measured = int_time / self.demodulation_frequency
            oversampling = sample_rate / (2 * self.demodulation_frequency)
            if samp_time > samples_time_max:
                int_time_max = samples_time_max / sample_rate
                Warning(
                    'int_time {} is longer than total_time - delay ({})'.format(int_time, int_time_max))
            elif oscilations_measured < 10:
                Warning(
                    '{} oscilations measured, recommend at least 10: '
                    'decrease sampling rate, take more samples or increase demodulation freq'.format(oscilations_measured))
            elif oversampling < 1:
                Warning('oversampling rate is {}, recommend > 1: '
                    'increase sampling rate or decrease demodulation frequency'.format(oversampling))
        else:
            samp_time = samples_per_record - samp_delay
        
        
        self.samples_time = samp_time
        self.samples_delay = samp_delay
        
        self.acquisition.update_acquisition_kwargs(samp_time, **kwargs)

    def pre_start_capture(self):
        """
        See AcquisitionController
        :return:
        """
        alazar = self._get_alazar()
        self.samples_per_record = alazar.samples_per_record.get()
        self.records_per_buffer = alazar.records_per_buffer.get()
        self.buffers_per_acquisition = alazar.buffers_per_acquisition.get()
        self.board_info = alazar.get_idn()
        self.buffer = np.zeros(self.samples_per_record *
                               self.records_per_buffer *
                               self.number_of_channels,
                               dtype=np.uint16)

        integer_list = np.arange(self.samples_per_record, dtype=np.uint16)
        angle_list = (2 * np.pi * self.demodulation_frequency /
                      self.sample_rate * integer_list)

        self.cos_list = np.cos(angle_list)
        self.sin_list = np.sin(angle_list)

    def pre_acquire(self):
        """
        See AcquisitionController
        :return:
        """
        # this could be used to start an Arbitrary Waveform Generator, etc...
        # using this method ensures that the contents are executed AFTER the
        # Alazar card starts listening for a trigger pulse
        pass

    def handle_buffer(self, data):
        """
        See AcquisitionController
        :return:
        """
        # average over buffers
        self.buffer += data

    def post_acquire(self):
        """
        See AcquisitionController
        :return:
        """
        records_per_acquisition = (1. * self.buffers_per_acquisition *
                                   self.records_per_buffer)
        # for ATS9360 samples are arranged in the buffer as follows:
        # S00A, S00B, S01A, S01B...S10A, S10B, S11A, S11B...
        # where SXYZ is record X, sample Y, channel Z.

        # break buffer up into records and averages over them
        records_per_acquisition = (self.buffers_per_acquisition *
                                   self.records_per_buffer)
        recordA = np.zeros(self.samples_per_record, dtype=np.uint16)
        for i in range(self.records_per_buffer):
            i0 = (i * self.samples_per_record * self.number_of_channels)
            i1 = (i0 + self.samples_per_record * self.number_of_channels)
            recordA += np.uint16(self.buffer[i0:i1:self.number_of_channels] /
                        records_per_acquisition)

        # do demodulation
        magA, phaseA = self.fit(recordA)

        # same for chan b
        if self.chan_b:
            recordB = np.zeros(self.samples_per_record, dtype=np.uint16)
            for i in range(self.records_per_buffer):
                i0 = (i * self.samples_per_record *
                      self.number_of_channels + 1)
                i1 = (i0 + self.samples_per_record * self.number_of_channels)
                recordB += np.uint16(self.buffer[i0:i1:self.number_of_channels] /
                            records_per_acquisition)
            magB, phaseB = self.fit(recordB)

        return magA, phaseA

    def fit(self, rec):
        # convert rec to volts
        bps = self.board_info['bits_per_sample']
        if bps == 12:
            volt_rec = sample_to_volt_u12(rec, bps)
        else:
            Warning('sample to volt conversion does not exist for'
                    ' bps != 12, centered raw samples returned')
            volt_rec = rec - np.mean(rec)

        # multiply with software wave
        re_wave = np.multiply(volt_rec, self.cos_list)
        im_wave = np.multiply(volt_rec, self.sin_list)
        cutoff = self.demodulation_frequency / 2

        # filter out higher freq component
        if self.filter == 0:
            re_filtered = filter_win(re_wave, cutoff,
                                           self.sample_rate, self.numtaps)
            im_filtered = filter_win(re_wave, cutoff,
                                           self.sample_rate, self.numtaps)
        elif self.filter == 1:
            re_filtered = filter_ham(re_wave, cutoff,
                                           self.sample_rate, self.numtaps)
            im_filtered = filter_ham(im_wave, cutoff,
                                           self.sample_rate, self.numtaps)
        elif self.filter == 2:
            re_filtered = filter_ls(re_wave, cutoff,
                                          self.sample_rate, self.numtaps)
            im_filtered = filter_ls(im_wave, cutoff,
                                          self.sample_rate, self.numtaps)

        # apply int limits
        start = self.samples_delay
        if self.int_time:
            end = self.int_time * self.sample_rate + start
        else:
            end = None
        re_limited = re_filtered[start:end]
        im_limited = im_filtered[start:end]

        # convert to magnitude and phase
        complex_num = re_limited + im_limited * 1j
        mag = abs(complex_num)
        phase = np.angle(complex_num, deg=True)

        return mag, phase
        
def filter_win(rec, cutoff, sample_rate, numtaps):
    nyq_rate = sample_rate / 2.
    fir_coef = signal.firwin(numtaps, cutoff / nyq_rate)
    filtered_rec = signal.lfilter(fir_coef, 1.0, rec)
    return filtered_rec

def sample_to_volt_u12(raw_samples, bps):
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