import logging
from .ATS import AcquisitionController
from .ATS9360 import AlazarTech_ATS9360
import numpy as np
from qcodes import Parameter
import qcodes.instrument_drivers.AlazarTech.acq_helpers as helpers
from qcodes.instrument.parameter import ManualParameter


class AcqVariablesParam(Parameter):
    """
    Parameter of an AcquisitionController which has a _check_and_update_instr
    function used for validation and to update instrument attributes and a
    _get_default function which it uses to set the parameter to an instrument
    caluclated default.
    """
    def __init__(self, name, instrument, check_and_update_fn,
                 default_fn, initial_value=None):
        super().__init__(name)
        self._instrument = instrument
        self._save_val(initial_value)
        setattr(self, '_check_and_update_instr', check_and_update_fn)
        setattr(self, '_get_default', default_fn)

    def set(self, value):
        self._check_and_update_instr(value)
        self._save_val(value)

    def get(self):
        return self._latest()['value']

    def to_default(self):
        default = self._get_default()
        self.set(default)

    def check(self):
        val = self._latest()['value']
        self._check_and_update_instr(val)
        return True


class RecordsAcqParam(Parameter):
    """
    Software controlled parameter class for Alazar acquisition. To be used with
    HD_Records_Controller (tested with ATS9360 board) for return of an array of
    records data from the Alazar, averaged over and buffers.

    TODO(nataliejpg) setpoints (including names and units)
    TODO(nataliejpg) setpoint units
    TODO(nataliejpg) convert demod_index setpoint into actual frequency
    TODO(nataliejpg) convert records setpoint into actual frequency
    """

    def __init__(self, name, instrument, demod_length):
        super().__init__(name)
        self._instrument = instrument
        self.acquisition_kwargs = {}
        self.names = ('magnitude', 'phase')
        self.shapes = ((demod_length, ), (demod_length, ))

    def update_sweep(self, npts, start=None, stop=None):
        demod_length = self._instrument._demod_length
        # rec_list = tuple(np.linspace(start, stop, num=npts))
        # demod_index = tuple(range(demod_length))
        if demod_length > 1:
            self.shapes = ((demod_length, npts), (demod_length, npts))
        else:
            self.shapes = ((npts,), (npts,))

    def get(self):
        mag, phase = self._instrument._get_alazar().acquire(
            acquisition_controller=self._instrument,
            **self.acquisition_kwargs)
        return mag, phase


class HD_Records_Controller(AcquisitionController):
    """
    This is the Acquisition Controller class which works with the ATS9360,
    averaging over buffers and samples, demodulating with software
    reference signals.

    Args:
        name: name for this acquisition_conroller as an instrument
        alazar_name: the name of the alazar instrument such that this
            controller can communicate with the Alazar
        demod_freqs: the frequency of the software wave to be created
        filter: the filter to be used to filter out double freq component
            ('win' - window, 'ls' - least squared, 'ave' - averaging)
        numtaps: number of freq components used in filter
        chan_b: whether there is also a second channel of data to be processed
            and returned
        **kwargs: kwargs are forwarded to the Instrument base class

    TODO(nataliejpg) test filter options
    TODO(nataliejpg) finish implementation of channel b option
    TODO(nataliejpg) what should be private?
    TODO(nataliejpg) where should filter_dict live?
    TODO(nataliejpg) demod_freq should be changeable number: maybe channels
    TODO(nataliejpg) try using fit from helpers
    TODO(nataliejpg) make records a parameter or similar (MultiParam?)
    TODO(nataliejpg) check docstrings
    """

    filter_dict = {'win': 0, 'ls': 1, 'ave': 2}
    samples_divisor = AlazarTech_ATS9360.samples_divisor

    def __init__(self, name, alazar_name, demod_length=1, filter='win',
                 numtaps=101, chan_b=False, **kwargs):
        self.filter_settings = {'filter': self.filter_dict[filter],
                                'numtaps': numtaps}
        self.chan_b = chan_b
        self._demod_length = demod_length
        self.number_of_channels = 2
        self.samples_per_record = None
        self.records_per_buffer = None
        self.sample_rate = None
        super().__init__(name, alazar_name, **kwargs)

        self.add_parameter(name='acquisition',
                           demod_length=self._demod_length,
                           parameter_class=RecordsAcqParam)
        for i in range(demod_length):
            self.add_parameter(name='demod_freq_{}'.format(i),
                               parameter_class=ManualParameter)
        self.add_parameter(name='int_time',
                           check_and_update_fn=self._update_int_time,
                           default_fn=self._int_time_default,
                           parameter_class=AcqVariablesParam)
        self.add_parameter(name='int_delay',
                           check_and_update_fn=self._update_int_delay,
                           default_fn=self._int_delay_default,
                           parameter_class=AcqVariablesParam)

    def _update_int_time(instr, value):
        """
        Function to validate value for int_time before setting parameter
        value
        Checks: limit between 0 and 0.1s
                acq knows sample_rate (doesn't check with alazar for accuracy)
                number of oscilation measured in this time
                oversampling rate
        Sets: samples_per_record of acq controller
              acquisition_kwarg['samples_per_record'] of acquisition param
              shape of acquisition param
        """
        if (value is None) or not (0 <= value <= 0.1):
            raise ValueError('int_time must be 0 <= value <= 1')
        alazar = instr._get_alazar()
        instr.sample_rate = alazar.get_sample_rate()
        if instr.get_max_demod_freq() is not None:
            oscilations_measured = value * instr.get_max_demod_freq()
            oversampling = instr.sample_rate / (2 * instr.get_max_demod_freq())
            if oscilations_measured < 10:
                logging.warning('{} oscilations measured, recommend at '
                                'least 10: decrease sampling rate, take '
                                'more samples or increase demodulation '
                                'freq'.format(oscilations_measured))
            elif oversampling < 1:
                logging.warning('oversampling rate is {}, recommend > 1: '
                                'increase sampling rate or decrease '
                                'demodulation frequency'.format(oversampling))
        if instr.int_delay() is None:
            instr.int_delay.to_default()

        # update acquision kwargs and acq controller value
        total_time = value + instr.int_delay()
        samples_needed = total_time * instr.sample_rate
        instr.samples_per_record = helpers.roundup(
            samples_needed, instr.samples_divisor)
        instr.acquisition.acquisition_kwargs.update(
            samples_per_record=instr.samples_per_record)

    def _update_int_delay(instr, value):
        """
        Function to validate value for int_delay before setting parameter
        value
        Checks: limit between 0 and 1s
                acq knows sample_rate (doesn't check with alazar for accuracy)
                number of samples discarded >= numtaps
        Sets: samples_per_record of acq controller
              acquisition_kwarg['samples_per_record'] of acquisition param
              shape of acquisition param
        """
        if (value is None) or not (0 <= value <= 0.1):
            raise ValueError('int_delay must be 0 <= value <= 1')
        alazar = instr._get_alazar()
        instr.sample_rate = alazar.get_sample_rate()
        samples_delay_min = (instr.filter_settings['numtaps'] - 1)
        int_delay_min = samples_delay_min / instr.sample_rate
        if value < int_delay_min:
            logging.warning(
                'delay is less than recommended for filter choice: '
                '(expect delay >= {})'.format(int_delay_min))

        # update acquision kwargs and acq controller value
        total_time = value + (instr.int_time() or 0)
        samples_needed = total_time * instr.sample_rate
        instr.samples_per_record = helpers.roundup(
            samples_needed, instr.samples_divisor)
        instr.acquisition.acquisition_kwargs.update(
            samples_per_record=instr.samples_per_record)

    def _int_delay_default(instr):
        """
        Returns minimum int_delay recommended for (numtaps - 1)
        samples to be discarded as recommended for filter
        """
        alazar = instr._get_alazar()
        instr.sample_rate = alazar.get_sample_rate()
        samp_delay = instr.filter_settings['numtaps'] - 1
        return samp_delay / instr.sample_rate

    def _int_time_default(instr):
        """
        Returns max total time for integration based on samples_per_record,
        sample_rate and int_delay
        """
        if instr.samples_per_record is (0 or None):
            raise ValueError('Cannot set int_time to max if acq controller'
                             ' has 0 or None samples_per_record, choose a '
                             'value for int_time and samples_per_record will '
                             'be set accordingly')
        alazar = instr._get_alazar()
        instr.sample_rate = alazar.get_sample_rate()
        total_time = ((instr.samples_per_record / instr.sample_rate) -
                      (instr.int_delay() or 0))
        return total_time

    def get_max_demod_freq(self):
        """
        Returns the largest demodulation frequency
        nb: really hacky and we should have channels in qcodes but we don't
        (at time of writing)
        """
        freqs = list(filter(None, [getattr(self, 'demod_freq_{}'.format(c))()
                                   for c in range(self._demod_length)]))
        if len(freqs) > 0:
            return max(freqs)
        else:
        return None

    def update_filter_settings(self, filter, numtaps):
        """
        Updates the settings of the filter for filtering out
        double frwuency component for demodulation.

        Args:
            filter (str): filter type ('win' or 'ls')
            numtaps (int): numtaps for filter
        """
        self.filter_settings.update({'filter': self.filter_dict[filter],
                                     'numtaps': numtaps})

    def update_acquisition_kwargs(self, **kwargs):
        """
        Updates the kwargs to be used when
        alazar_driver.acquisition() is called via a get call of the
        acquisition SamplesParam. Should be used by the user for 
        updating averaging settings since the 'samples_per_record'
        kwarg is updated via the int_time and int_delay parameters

        Kwargs (ints):
            records_per_buffer
            buffers_per_acquisition
            allocated_buffers
        """
        if 'samples_per_record' in kwargs:
            raise ValueError('With HD_Samples_Controller '
                             'samples_per_record cannot be set manually '
                             'via update_acquisition_kwargs and should instead'
                             ' be set by setting int_time and int_delay')
        if 'records_per_buffer' in kwargs:
            self.acquisition.update_sweep(kwargs['records_per_buffer'])
            self.records_per_buffer = kwargs['records_per_buffer']
        self.acquisition.acquisition_kwargs.update(**kwargs)

    def pre_start_capture(self):
        """
        Called before capture start to update Acquisition Controller with
        alazar acquisition params and set up software wave for demodulation.
        """
        alazar = self._get_alazar()
        if self.samples_per_record != alazar.samples_per_record.get():
            raise Exception('acq controller samples per record does not match'
                            ' instrument value, most likely need '
                            'to set and check int_time and int_delay')
        if self.sample_rate != alazar.get_sample_rate():
            raise Exception('acq controller sample rate does not match '
                            'instrument value, most likely need '
                            'to set and check int_time and int_delay')
        if self.records_per_buffer != alazar.records_per_buffer.get():
            raise Exception('acq controller records per buffer does not match '
                            'instrument value, most likely need '
                            'to call update_acquisition_kwargs with'
                            'records_per_buffer as a param')
        if self.get_max_demod_freq() is None:
            raise Exception('no demodulation frequencies set')
        self.buffers_per_acquisition = alazar.buffers_per_acquisition.get()
        self.board_info = alazar.get_idn()
        self.buffer = np.zeros(self.samples_per_record *
                               self.records_per_buffer *
                               self.number_of_channels)

        mat_shape = (self._demod_length, self.records_per_buffer,
                     self.samples_per_record)
        demod_list = np.array([getattr(self, 'demod_freq_{}'.format(n))()
                               for n in range(self._demod_length)])
        integer_list = np.arange(self.samples_per_record)
        integer_mat = np.outer(np.ones(self.records_per_buffer), integer_list)
        angle_mat = 2 * np.pi * \
            np.outer(demod_list, integer_mat).reshape(
                mat_shape) / self.sample_rate
        self.cos_mat = np.cos(angle_mat)
        self.sin_mat = np.sin(angle_mat)

    def pre_acquire(self):
        pass

    def handle_buffer(self, data):
        """
        Adds data from alazar to buffer (effectively averaging)
        """
        self.buffer += data

    def post_acquire(self):
        """
        Processes the data according to ATS9360 settings, splitting into
        records and then applying demodulation fit
        nb: currently only channel A

        Returns:
            magnitude (numpy array): shape = (demod_length, records_per_buffer)
            phase (numpy array): shape = (demod_length, records_per_buffer
        """
        # for ATS9360 samples are arranged in the buffer as follows:
        # S00A, S00B, S01A, S01B...S10A, S10B, S11A, S11B...
        # where SXYZ is record X, sample Y, channel Z.

        # break buffer up into records and shapes to be (records, samples)
        recordsA = np.empty((self.records_per_buffer, self.samples_per_record),
                            dtype=np.uint16)
        for i in range(self.records_per_buffer):
            i0 = (i * self.samples_per_record * self.number_of_channels)
            i1 = (i0 + self.samples_per_record * self.number_of_channels)
            recordsA[i, :] = np.uint16(
                self.buffer[i0:i1:self.number_of_channels] /
                self.buffers_per_acquisition)

        # do demodulation
        magA, phaseA = self._fit(recordsA)

        # same for chan b
        if self.chan_b:
            raise NotImplementedError('chan b code not complete')

        return magA, phaseA

    def _fit(self, rec):
        """
        Applies volts conversion, demodulation fit, low bandpass filter
        and integration limits to samples array

        Args:
            rec (numpy array): record from alazar to be multiplied
                               with the software signal, filtered and limited
                               to integration limits shape = (samples_taken, )

        Returns:
            magnitude (numpy array): shape = (demod_length, records_per_buffer)
            phase (numpy array): shape = (demod_length, records_per_buffer)
        """
        # convert rec to volts
        bps = self.board_info['bits_per_sample']
        if bps == 12:
            volt_rec = helpers.sample_to_volt_u12(rec, bps)
        else:
            logging.warning('sample to volt conversion does not exist for'
                            ' bps != 12, centered raw samples returned')
            volt_rec = rec - np.mean(rec, axis=1)

        # volt_rec to matrix and multiply with demodulation signal matrices
        mat_shape = (self._demod_length, self.records_per_buffer,
                     self.samples_per_record)
        volt_rec_mat = np.outer(
            np.ones(self._demod_length), volt_rec).reshape(mat_shape)
        re_mat = np.multiply(volt_rec_mat, self.cos_mat)
        im_mat = np.multiply(volt_rec_mat, self.sin_mat)

        # filter out higher freq component
        cutoff = self.get_max_demod_freq() / 10
        if self.filter_settings['filter'] == 0:
            re_filtered = helpers.filter_win(re_mat, cutoff,
                                             self.sample_rate,
                                             self.filter_settings['numtaps'],
                                             axis=2)
            im_filtered = helpers.filter_win(im_mat, cutoff,
                                             self.sample_rate,
                                             self.filter_settings['numtaps'],
                                             axis=2)
        elif self.filter_settings['filter'] == 1:
            re_filtered = helpers.filter_ls(re_mat, cutoff,
                                            self.sample_rate,
                                            self.filter_settings['numtaps'],
                                            axis=2)
            im_filtered = helpers.filter_ls(im_mat, cutoff,
                                            self.sample_rate,
                                            self.filter_settings['numtaps'],
                                            axis=2)
        elif self.filter_settings['filter'] == 2:
            re_filtered = re_mat
            im_filtered = im_mat

        # apply integration limits
        beginning = int(self.int_delay() * self.sample_rate)
        end = beginning + int(self.int_time() * self.sample_rate)

        re_limited = re_filtered[:, :, beginning:end]
        im_limited = im_filtered[:, :, beginning:end]

        # convert to magnitude and phase
        complex_mat = re_limited + im_limited * 1j
        magnitude = np.mean(abs(complex_mat), axis=2)
        phase = np.mean(np.angle(complex_mat, deg=True), axis=2)

        return magnitude, phase
