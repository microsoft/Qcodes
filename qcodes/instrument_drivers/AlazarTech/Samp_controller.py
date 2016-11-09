from .ATS import AcquisitionController
import numpy as np
from qcodes import Parameter
import acquistion_tools as tools


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
    TODO(nataliejpg) update docstrings :P
    """

    def __init__(self, name, alazar_name, demod_freq, samp_rate=500e6,
                 int_delay=None, int_time=None, filt='win',
                 numtaps=101, chan_b=False, **kwargs):
        filter_dict = {'win': 0, 'hamming': 1, 'ls': 2}
        self.demodulation_frequency = demod_freq
        self.sample_rate = samp_rate
        self.filter = filter_dict[filt]
        self.int_time = int_time
        self.int_delay = int_delay
        self.numtaps = numtaps
        self.chan_b = chan_b
        self.samples_per_record = 0
        self.records_per_buffer = 0
        self.buffers_per_acquisition = 0
        self.number_of_channels = 2
        self.samples_delay = None
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
        self.board_info = alazar.get_board_info(alazar.dll_path)
        self.buffer = np.zeros(self.samples_per_record *
                               self.records_per_buffer *
                               self.number_of_channels)

        integer_list = np.arange(self.samples_per_record)
        angle_list = (2 * np.pi * self.demodulation_frequency /
                      self.sample_rate * integer_list)

        self.cos_list = np.cos(angle_list)
        self.sin_list = np.sin(angle_list)

        if self.int_delay:
            self.samples_delay = self.int_delay * self.sample_rate
            if self.samples_delay < (self.numtaps - 1):
                expected_delay = (self.numtaps - 1) / self.sample_rate
                Warning(
                    'delay is less than recommended for filter choice: '
                    '(expect delay >= {}'.format(expected_delay))
        else:
            self.samples_delay = self.numtaps - 1

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
        recordA = np.zeros(self.samples_per_record)
        for i in range(self.records_per_buffer):
            i0 = (i * self.samples_per_record * self.number_of_channels)
            i1 = (i0 + self.samples_per_record * self.number_of_channels)
            recordA += (self.buffer[i0:i1:self.number_of_channels] /
                        records_per_acquisition)

        # do demodulation
        magA, phaseA = self.fit(recordA)

        # same for chan b
        if self.chan_b:
            recordB = np.zeros(self.samples_per_record)
            for i in range(self.records_per_buffer):
                i0 = (i * self.samples_per_record *
                      self.number_of_channels + 1)
                i1 = (i0 + self.samples_per_record * self.number_of_channels)
                recordB += (self.buffer[i0:i1:self.number_of_channels] /
                            records_per_acquisition)
            magB, phaseB = self.fit(recordB)

        return magA, phaseA

    def fit(self, rec):
        # convert rec to volts
        if self.board_info['bits_per_sample'] == 12:
            volt_rec = tools.sample_to_volt_u12(rec)
        else:
            Warning(
                'sample to volt conversion does not exist for '
                'bps != 12, raw samples centered and returned')
            volt_rec = rec - np.mean(rec)

        # multiply with software wave
        re_wave = np.multiply(volt_rec, self.cos_list)
        im_wave = np.multiply(volt_rec, self.sin_list)
        cutoff = self.demodulation_frequency / 2

        # filter out higher freq component
        if self.filter == 0:
            re_filtered = tools.filter_win(re_wave, cutoff,
                                           self.samp_rate, self.numtaps)
            im_filtered = tools.filter_win(re_wave, cutoff,
                                           self.samp_rate, self.numtaps)
        elif self.filter == 1:
            re_filtered = tools.filter_ham(re_wave, cutoff,
                                           self.samp_rate, self.numtaps)
            im_filtered = tools.filter_ham(im_wave, cutoff,
                                           self.samp_rate, self.numtaps)
        elif self.filter == 2:
            re_filtered = tools.filter_ls(re_wave, cutoff,
                                          self.samp_rate, self.numtaps)
            im_filtered = tools.filter_ls(im_wave, cutoff,
                                          self.samp_rate, self.numtaps)

        # apply int limits
        start = self.samples_delay
        end = self.int_time * self.samp_rate + start
        re_limited = re_filtered[start:end]
        im_limited = im_filtered[start:end]

        # convert to magnitude and phase
        complex_num = re_limited + im_limited * 1j
        mag = abs(complex_num)
        phase = np.angle(complex_num, deg=True)

        return mag, phase
