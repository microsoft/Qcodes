from .ATS import AcquisitionController
import numpy as np
from scipy import signal
from qcodes import Parameter


class RecordsParam(Parameter):
    """
    Hardware controlled parameter class for Alazar acquisition. To be used with
    HD_Records_Controller (tested with ATS9360 board) for return of an array of
    record data from the Alazar, averaged over samples and buffers.
    """
    def __init__(self, name, instrument):
        super().__init__(name)
        self._instrument = instrument
        self.acquisitionkwargs = {}
        self.names = ('magnitude', 'phase')
        self.units = ('', '')
        self.setpoint_names = (('record_num',), ('record_num',))
        self.setpoints = ((1,), (1,))
        self.shapes = ((1,), (1,))

    def update_acquisition_kwargs(self, **kwargs):
        # needed to update config of the software parameter on sweep change
        # freq setpoints tuple as needs to be hashable for look up
        if 'records_per_buffer' in kwargs:
            npts = kwargs['records_per_buffer']
            n = tuple(np.arange(npts))
            self.setpoints = ((n,), (n,))
            self.shapes = ((npts,), (npts,))
        else:
            raise ValueError('records_per_buffer must be specified')
        # updates dict to be used in acquisition get call
        self.acquisitionkwargs.update(**kwargs)

    def get(self):
        mag, phase = self._instrument._get_alazar().acquire(
            acquisition_controller=self._instrument,
            **self.acquisitionkwargs)
        return mag, phase


class HD_Records_Controller(AcquisitionController):
    """
    seq 0
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
    TODO(nataliejpg) add filter options
    TODO(nataliejpg) test mag phase logic
    TODO(nataliejpg) implement record B thinking
    """

    def __init__(self, name, alazar_name, demod_freq, samp_rate=500e6,
                 filt='win', chan_b=False, **kwargs):
        filter_dict = {'win': 0, 'hamming': 1, 'ls': 2}
        self.demodulation_frequency = demod_freq
        self.sample_rate = samp_rate
        self.filter = filter_dict[filt]
        #self.chan_b = chan_b
        self.sample_rate = samp_rate
        self.samples_per_record = 0
        self.records_per_buffer = 0
        self.buffers_per_acquisition = 0
        self.number_of_channels = 2
        self.cos_mat = None
        self.sin_mat = None
        self.buffer = None
        # make a call to the parent class and by extension,
        # create the parameter structure of this class
        super().__init__(name, alazar_name, **kwargs)

        self.add_parameter(name='acquisition',
                           parameter_class=RecordsParam)

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
                      self.sample_rate * integer_list)

        cos_list = np.cos(angle_list).reshape(self.samples_per_record, 1)
        sin_list = np.sin(angle_list).reshape(self.samples_per_record, 1)
        self.cos_mat = np.kron(np.ones((1, self.records_per_buffer)), cos_list)
        self.sin_mat = np.kron(np.ones((1, self.records_per_buffer)), sin_list)

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
        self.buffer += data

    def post_acquire(self):
        """
        See AcquisitionController
        :return:
        """

        # for ATS9360 samples are arranged in the buffer as follows:
        # S00A, S00B, S01A, S01B...S10A, S10B, S11A, S11B...
        # where SXYZ is record X, sample Y, channel Z.

        # reshapes date to be (samples * records)
        recordA = np.zeros((self.samples_per_record, self.records_per_buffer))
        # recordB = np.zeros((self.samples_per_record, self.records_per_buffer))\
        for i in range(self.records_per_buffer):
            i0 = i * self.number_of_channels * self.samples_per_record
            i1 = i0 + self.number_of_channels * self.samples_per_record
            recordA[:, i] = (self.buffer[i0:i1:self.number_of_channels] /
                             self.buffers_per_acquisition)
            # recordB[:, i] = self.buffer[1:full_rec_length:step] / buffers
        # return averaged chan A data (records)
        magA, phaseA = self.fit(recordA)
        
        return magA, phaseA

    def fit(self, rec):
        # center rec around 0
        rec = rec - np.mean(rec)

        # multiply with software wave
        re_wave = np.multiply(rec, self.cos_mat)
        im_wave = np.multiply(rec, self.sin_mat)
        cutoff = self.demodulation_frequency
        numtaps = 30
        axis = 0

        # filter out double freq component to obtian constant term
        if self.filter == 0:
            RePart = self.filter_win(re_wave, numtaps, cutoff, axis=axis)
            ImPart = self.filter_win(im_wave, numtaps, cutoff, axis=axis)
        elif self.filter == 1:
            RePart = self.filter_hamming(re_wave, numtaps, cutoff, axis=axis)
            ImPart = self.filter_hamming(im_wave, numtaps, cutoff, axis=axis)
        elif self.filter == 2:
            RePart = self.filter_ls(re_wave, numtaps, cutoff, axis, axis=axis)
            ImPart = self.filter_ls(im_wave, numtaps, cutoff, axis, axis=axis)

        # convert to magnitude and phase data and average over samples
        # data returned is (records)
        complex_mat = RePart + ImPart * 1j
        mag = np.mean(2 * abs(complex_mat), axis=0)
        phase = np.mean(np.angle(complex_mat, deg=True), axis=0)

        return mag, phase

    def filter_hamming(self, rec, numtaps, cutoff, axis=-1):
        sample_rate = self.sample_rate
        nyq_rate = sample_rate / 2.
        fir_coef = signal.firwin(numtaps,
                                 cutoff / nyq_rate,
                                 window="hamming")
        filtered_rec = 2 * signal.lfilter(fir_coef, 1.0, rec, axis=axis)
        return filtered_rec

    def filter_win(self, rec, numtaps, cutoff, axis=-1):
        sample_rate = self.sample_rate
        nyq_rate = sample_rate / 2.
        fir_coef = signal.firwin(numtaps,
                                 cutoff / nyq_rate)
        filtered_rec = 2 * signal.lfilter(fir_coef, 1.0, rec, axis=axis)
        return filtered_rec

    def filter_ls(self, rec, numtaps, cutoff, axis=-1):
        sample_rate = self.sample_rate
        nyq_rate = sample_rate / 2.
        bands = [0, cutoff / nyq_rate, cutoff / nyq_rate, 1]
        desired = [1, 1, 0, 0]
        fir_coef = signal.firls(numtaps,
                                bands,
                                desired,
                                nyq=nyq_rate)
        filtered_rec = 2 * signal.lfilter(fir_coef, 1.0, rec, axis=axis)
        return filtered_rec
