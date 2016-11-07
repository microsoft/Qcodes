from .ATS import AcquisitionController
import numpy as np
from qcodes import Parameter
from scipy import signal


class SamplesParam(Parameter):
    """
    Hardware controlled parameter class for Alazar acquisition. To be used with
    HD_Samples_Controller (tested with ATS9360 board) for return of an array of
    sample data from the Alazar, averaged over records and buffers.
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

    def update_acquisition_kwargs(self, **kwargs):
        # needed to update config of the software parameter on sweep change
        # freq setpoints tuple as needs to be hashable for look up
        if 'samples_per_record' in kwargs:
            npts = kwargs['samples_per_record']
            n = tuple(np.arange(npts))
            self.setpoints = ((n,), (n,))
            self.shapes = ((npts,), (npts,))
        else:
            raise ValueError('samples_per_record must be specified')
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
    TODO(nataliejpg) make filter settings not hard coded
    """

    def __init__(self, name, alazar_name, demod_freq, samp_rate=500e6,
                 filt='win', chan_b=False, **kwargs):
        filter_dict = {'win': 0, 'hamming': 1, 'ls': 2}
        self.demodulation_frequency = demod_freq
        self.sample_rate = samp_rate
        self.filter = filter_dict[filt]
        self.chan_b = chan_b
        self.samples_per_record = 0
        self.records_per_buffer = 0
        self.buffers_per_acquisition = 0
        self.number_of_channels = 2
        self.cos_list = None
        self.sin_list = None
        self.buffer = None
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
        self.acquisition.update_acquisition_kwargs(**kwargs)

    def pre_start_capture(self):
        """
        See AcquisitionController
        :return:
        """
        alazar = self._get_alazar()
        self.samples_per_record = alazar.samples_per_record.get()
        self.records_per_buffer = alazar.records_per_buffer.get()
        self.buffers_per_acquisition = alazar.buffers_per_acquisition.get()
        self.buffer = np.zeros(self.samples_per_record *
                               self.records_per_buffer *
                               self.number_of_channels)

        integer_list = np.arange(self.samples_per_record)
        angle_list = (2 * np.pi * self.demodulation_frequency /
                      self.sample_speed * integer_list)

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

        # breaks buffer up into records and averages over them
        # leaving data in form (samples)
        records_per_acquisition = (self.buffers_per_acquisition *
                                   self.records_per_buffer)
        recordA = np.zeros(self.samples_per_record)
        for i in range(self.records_per_buffer):
            i0 = (i * self.samples_per_record * self.number_of_channels)
            i1 = (i0 + self.samples_per_record * self.number_of_channels)
            recordA += (self.buffer[i0:i1:self.number_of_channels] /
                        records_per_acquisition)

        # demodulate record, data returned (samples)
        magA, phaseA = self.fit(recordA)

        if self.chan_b:
            recordB = np.zeros(self.samples_per_record)
            for i in range(self.records_per_buffer):
                i0 = (i * self.samples_per_record * self.number_of_channels + 1)
                i1 = (i0 + self.samples_per_record * self.number_of_channels)
                recordB += (self.buffer[i0:i1:self.number_of_channels] /
                            records_per_acquisition)
            magB, phaseB = self.fit(recordB)

        return magA, phaseA

    def fit(self, rec):
        # center rec around 0
        rec = rec - np.mean(rec)

        # multiply with software wave
        re_wave = np.multiply(rec, self.cos_list)
        im_wave = np.multiply(rec, self.sin_list)
        cutoff = self.demodulation_frequency
        numtaps = 30

        # filter out higher freq component
        if self.filter == 0:
            RePart = self.filter_win(re_wave, numtaps, cutoff)
            ImPart = self.filter_win(im_wave, numtaps, cutoff)
        elif self.filter == 1:
            RePart = self.filter_hamming(re_wave, numtaps, cutoff)
            ImPart = self.filter_hamming(im_wave, numtaps, cutoff)
        elif self.filter == 2:
            RePart = self.filter_ls(re_wave, numtaps, cutoff)
            ImPart = self.filter_ls(im_wave, numtaps, cutoff)

        # convert to magnitude and phase
        # data returnded (samples)
        complex_num = RePart + ImPart * 1j
        mag = 2 * abs(complex_num)
        phase = np.angle(complex_num, deg=True)

        return mag, phase

    def filter_hamming(self, rec, numtaps, cutoff):
        sample_rate = self.sample_rate
        nyq_rate = sample_rate / 2.
        fir_coef = signal.firwin(numtaps,
                                 cutoff / nyq_rate,
                                 window="hamming")
        filtered_rec = 2 * signal.lfilter(fir_coef, 1.0, rec)
        return filtered_rec

    def filter_win(self, rec, numtaps, cutoff):
        sample_rate = self.sample_rate
        nyq_rate = sample_rate / 2.
        fir_coef = signal.firwin(numtaps,
                                 cutoff / nyq_rate)
        filtered_rec = 2 * signal.lfilter(fir_coef, 1.0, rec)
        return filtered_rec

    def filter_ls(self, rec, numtaps, cutoff):
        sample_rate = self.sample_rate
        nyq_rate = sample_rate / 2.
        bands = [0, cutoff / nyq_rate, cutoff / nyq_rate, 1]
        desired = [1, 1, 0, 0]
        fir_coef = signal.firls(numtaps,
                                bands,
                                desired,
                                nyq=nyq_rate)
        filtered_rec = 2 * signal.lfilter(fir_coef, 1.0, rec)
        return filtered_rec
