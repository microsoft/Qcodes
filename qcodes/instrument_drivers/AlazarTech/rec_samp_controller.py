import logging
from .ATS import AcquisitionController
import numpy as np
from qcodes import Parameter, MultiParameter
import qcodes.instrument_drivers.AlazarTech.acq_helpers as helpers


class AcqVariablesParam(Parameter):
    """
    Parameter of an AcquisitionController which has a _check_and_update_instr
    function used for validation and to update instrument attributes and a
    _get_default function which it uses to set the AcqVariablesParam to an
    instrument caluclated default.

    Args:
        name: name for this parameter
        instrument: acquisition controller instrument this parameter belongs to
        check_and_update_fn: instrument function to be used for value
            validation and updating instrument values
        default_fn (optional): instrument function to be used to calculate
            a default value to set parameter to
        initial_value (optional): initial value for parameter
    """

    def __init__(self, name, instrument, check_and_update_fn,
                 default_fn=None, initial_value=None):
        super().__init__(name)
        self._instrument = instrument
        self._save_val(initial_value)
        setattr(self, '_check_and_update_instr', check_and_update_fn)
        if default_fn is not None:
            setattr(self, '_get_default', default_fn)

    def set(self, value):
        """
        Function which checks value using validation function and then sets
        the Parameter value to this value.

        Args:
            value: value to set the parameter to
        """
        self._check_and_update_instr(value, param_name=self.name)
        self._save_val(value)

    def get(self):
        return self._latest()['value']

    def to_default(self):
        """
        Function which executes the default_fn specified to calculate the
        default value based on instrument values and then calls the set
        function with this value
        """
        try:
            default = self._get_default()
        except AttributeError as e:
            raise AttributeError('no default function for {} Parameter '
                                 '{}'.format(self.name, e))
        self.set(default)

    def check(self):
        """
        Function which checks the current Parameter value using the specified
        check_and_update_fn which can also serve to update instrument values.

        Return:
            True (if no errors raised when check_and_update_fn executed)
        """
        val = self._latest()['value']
        self._check_and_update_instr(val, param_name=self.name)
        return True


class RecordsSamplesAcqParam(MultiParameter):
    """
    TODO

    Args:
        name: name for this parameter
        instrument: acquisition controller instrument this parameter belongs to
        demod_length: numer of demodulation frequencies the acq controller has
            NB this is currently set in acq controller init but that's really
            not nice and should be changed when possible

    TODO(nataliejpg) setpoints (including names and units)
    TODO(nataliejpg) convert records setpoint into actual frequency
    """

    def __init__(self, name, instrument):
        names = ('magnitude', 'phase')
        shapes = ((), ())
        super().__init__(name, names, shapes, instrument)
        self.acquisition_kwargs = {}


    def update_time_sweep(self, time_npts, time_start, time_stop):
        """
        Function which updates the shape of the parameter (and it's setpoints
        when this is fixed)

        Args:
            time_npts: number of samples returned after processing
            time_start: start time of samples returned after processing
            time_stop: stop time of samples returned after processing
        """
        demod_length = self._instrument._demod_length
        # self._time_list = tuple(np.linspace(time_start,
        #                                     time_stop, num=time_npts))
        self._time_npts = time_npts
        if demod_length > 1:
            # demod_freqs = self._instrument.get_demod_freqs()
            # self.setpoints = ((demod_freqs, self._rec_list, self._time_list),
            #                   (demod_freqs, self._rec_list, self._time_list))
            self.shapes = ((demod_length, self._rec_npts, self._time_npts),
                           (demod_length, self._rec_npts, self._time_npts))
        else:
            self.shapes = ((self._rec_npts, self._time_npts),
                           (self._rec_npts, self._time_npts))
            # self.setpoints = ((self._rec_list, self._time_list),
            #                   (self._rec_list, self._time_list))

    def update_rec_sweep(self, rec_npts, rec_start=None, rec_stop=None):
        """
        Function which updates the shape of the parameter (and it's setpoints
        when this is fixed)

        Args:
            rec_npts: number of records returned
            rec_start (optional): start value of records returned
            rec_stop (optional): stop value of records returned
        """
        demod_length = self._instrument._demod_length
        # self._rec_list = tuple(np.linspace(rec_start,
        #                                    rec_stop, num=rec_npts))
        self._rec_npts = rec_npts
        if demod_length > 1:
            # demod_freqs = self._instrument.get_demod_freqs()
            # self.setpoints = ((demod_freqs, self._rec_list, self._time_list),
            #                   (demod_freqs, self._rec_list, self._time_list))
            self.shapes = ((demod_length, self._rec_npts, self._time_npts),
                           (demod_length, self._rec_npts, self._time_npts))
        else:
            self.shapes = ((self._rec_npts, self._time_npts),
                           (self._rec_npts, self._time_npts))
            # self.setpoints = ((self._rec_list, self._time_list),
            #                   (self._rec_list, self._time_list))

    def update_demod_setpoints(self, demod_freqs):
        """
        Function to update the demodulation frequency setpoints to be called
        when a demod_freq Parameter of the acq controller is updated

        Args:
            demod_freqs: numpy array of demodulation frequencies to use as
                setpoints if length > 1
        """
        demod_length = self._instrument._demod_length
        if demod_length > 1:
            pass
            # self.setpoints = ((demod_freqs, self._rec_list, self._time_list),
            #                   (demod_freqs, self._rec_list, self._time_list))
        else:
            pass

    def get(self):
        """
        Gets the magnitude and phase signal by calling acquire
        on the alazar (which in turn calls the processing functions of the
        aqcuisition controller before returning the processed data
        demodulated at specified frequencies and averaged over buffers)

        Returns:
            mag a numpy array of magnitude, shape (demod_length, records)
            phase a numpy array of phase, shape (demod_length, records)
        """
        mag, phase = self._instrument._get_alazar().acquire(
            acquisition_controller=self._instrument,
            **self.acquisition_kwargs)
        return mag, phase


class HD_RecordsSamples_Controller(AcquisitionController):
    """
    This is the Acquisition Controller class which works with the ATS9360,
    averaging over buffers and demodulating with software reference signal(s).

    Args:
        name: name for this acquisition_conroller as an instrument
        alazar_name: name of the alazar instrument such that this
            controller can communicate with the Alazar
        demod_length (default 1): number of demodulation frequencies
        filter (default 'win'): filter to be used to filter out double freq
            component ('win' - window, 'ls' - least squared, 'ave' - averaging)
        numtaps (default 101): number of freq components used in filter
        chan_b (default False): whether there is also a second channel of data
            to be processed and returned
        **kwargs: kwargs are forwarded to the Instrument base class

    TODO(nataliejpg) test filter options
    TODO(nataliejpg) finish implementation of channel b option
    TODO(nataliejpg) what should be private?
    TODO(nataliejpg) where should filter_dict live?
    TODO(nataliejpg) demod_freq should be changeable number: maybe channels
    TODO(nataliejpg) make record param 'start' and 'stop' meaningful
    """

    filter_dict = {'win': 0, 'ls': 1, 'ave': 2}

    def __init__(self, name, alazar_name, demod_length=1, filter='win',
                 numtaps=101, chan_b=False, **kwargs):
        self.filter_settings = {'filter': self.filter_dict[filter],
                                'numtaps': numtaps}
        self.chan_b = chan_b
        self._demod_length = demod_length
        self.number_of_channels = 2
        self.samples_per_record = None
        self.sample_rate = None
        super().__init__(name, alazar_name, **kwargs)

        self.add_parameter(name='acquisition',
                           parameter_class=RecordsSamplesAcqParam)
        for i in range(demod_length):
            self.add_parameter(name='demod_freq_{}'.format(i),
                               check_and_update_fn=self._update_demod_freq,
                               parameter_class=AcqVariablesParam)
        self.add_parameter(name='int_time',
                           check_and_update_fn=self._update_int_time,
                           default_fn=self._int_time_default,
                           parameter_class=AcqVariablesParam)
        self.add_parameter(name='int_delay',
                           check_and_update_fn=self._update_int_delay,
                           default_fn=self._int_delay_default,
                           parameter_class=AcqVariablesParam)
        self.add_parameter(name='record_num',
                           check_and_update_fn=self._update_rec_num,
                           parameter_class=AcqVariablesParam)

        self.samples_divisor = self._get_alazar().samples_divisor

    def _update_demod_freq(instr, value, param_name=None):
        """
        Function to validate and update acquisiton parameter when
        a demod_freq_ Parameter is changed

        Args:
            value to update demodulation frequency to

        Kwargs:
            param_name: used to update demod_freq list used for updating
                septionts of acquisition parameter

        Checks:
            1e6 <= value <= 500e6
            number of oscilation measured using current int_tiume param value
                at this demod frequency value
            oversampling rate for this demodulation frequency

        Sets:
            sample_rate attr of acq controller to be that of alazar
            setpoints of acquisiton parameter
        """
        if (value is None) or not (1e6 <= value <= 500e6):
            raise ValueError('demod_freqs must be 1e6 <= value <= 500e6')
        alazar = instr._get_alazar()
        instr.sample_rate = alazar.get_sample_rate()
        min_oscilations_measured = instr.int_time() * value
        oversampling = instr.sample_rate / (2 * value)
        if min_oscilations_measured < 10:
            logging.warning('{} oscilations measured for largest '
                            'demod freq, recommend at least 10: '
                            'decrease sampling rate, take '
                            'more samples or increase demodulation '
                            'freq'.format(min_oscilations_measured))
        elif oversampling < 1:
            logging.warning('oversampling rate is {}, recommend > 1: '
                            'increase sampling rate or decrease '
                            'demodulation frequency'.format(oversampling))
        demod_freqs = instr.get_demod_freqs()
        current_demod_index = ([int(s) for s in param_name.split()
                                if s.isdigit()][0])
        demod_freqs[current_demod_index] = value
        instr.acquisition.update_demod_setpoints(demod_freqs)

    def _update_record_num(instr, value, **kwargs):
        """
        Function to update the record number, and instr attributes. Main
        reason to record_num as a parameter is so we can more easily push the
        record number setting up to other instruments which will control this.

        Args:
            value to be validated and used for instrument attribute update

        Checks:
            1 <= value <= 1000

        Sets:
            acquisition_kwarg['records_per_buffer'] of acquisition param
            setpoints of acquisiton param
            shape of acquisition param
        """
        if (value is None) or not (1 <= value <= 1000):
            raise ValueError('int_time must be 1 <= value <= 1000')

        # update acquisition parameter shapes
        instr.acquisition.update_rec_sweep(value)

        # update acquision kwargs
        instr.acquisition.acquisition_kwargs.update(
            records_per_buffer=value)

    def _update_int_time(instr, value, **kwargs):
        """
        Function to validate value for int_time before setting parameter
        value and update instr attributes.

        Args:
            value to be validated and used for instrument attribute update

        Checks:
            0 <= value <= 0.1 seconds
            number of oscilation measured in this time
            oversampling rate

        Sets:
            sample_rate attr of acq controller to be that of alazar
            samples_per_record of acq controller
            acquisition_kwarg['samples_per_record'] of acquisition param
            setpoints of acquisiton param
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

        # update acquisition parameter shapes
        start = instr.int_delay()
        stop = start + value
        npts = int(value * instr.sample_rate)
        instr.acquisition.update_time_sweep(start, stop, npts)

        # update acquision kwargs and acq controller value
        total_time = value + instr.int_delay()
        samples_needed = total_time * instr.sample_rate
        instr.samples_per_record = helpers.roundup(
            samples_needed, instr.samples_divisor)
        instr.acquisition.acquisition_kwargs.update(
            samples_per_record=instr.samples_per_record)

    def _update_int_delay(instr, value, **kwargs):
        """
        Function to validate value for int_delay before setting parameter
        value and update instr attributes.

        Args:
            value to be validated and used for instrument attribute update

        Checks:
            0 <= value <= 0.1 seconds
            number of samples discarded >= numtaps

        Sets:
            sample_rate attr of acq controller to be that of alazar
            samples_per_record of acq controller
            acquisition_kwarg['samples_per_record'] of acquisition param
            setpoints of acquisiton param
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

        # update acquisition parameter shapes
        start = value
        stop = start + (instr.int_time() or 0)
        npts = int((instr.int_time() or 0) * instr.sample_rate)
        instr.acquisition.update_time_sweep(start, stop, npts)

        # update acquision kwargs and acq controller value
        total_time = value + (instr.int_time() or 0)
        samples_needed = total_time * instr.sample_rate
        instr.samples_per_record = helpers.roundup(
            samples_needed, instr.samples_divisor)
        instr.acquisition.acquisition_kwargs.update(
            samples_per_record=instr.samples_per_record)

    def _int_delay_default(instr):
        """
        Function to generate default int_delay value

        Returns:
            minimum int_delay recommended for (numtaps - 1)
            samples to be discarded as recommended for filter
        """
        alazar = instr._get_alazar()
        instr.sample_rate = alazar.get_sample_rate()
        samp_delay = instr.filter_settings['numtaps'] - 1
        return samp_delay / instr.sample_rate

    def _int_time_default(instr):
        """
        Function to generate defult int_time value

        Returns:
            max total time for integration based on samples_per_record,
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

    def get_demod_freqs(self):
        """
        Function to get all the demod_freq parameter values in a list, v hacky

        Returns:
            freqs a numpy array of demodulation frequencies
        """
        freqs = list(filter(None, [getattr(self, 'demod_freq_{}'.format(c))()
                                   for c in range(self._demod_length)]))
        return np.array(freqs)

    def get_max_demod_freq(self):
        """
        Returns:
            the largest demodulation frequency

        nb: really hacky and we should have channels in qcodes but we don't
        (at time of writing)
        """
        freqs = self.get_demod_freqs()
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
        acquisition RecordsSamplesAcqParam. Should be used by the user for
        updating averaging settings since the 'samples_per_record'
        and 'records_per_buffer' kwargs are updated via the int_time,
        int_delay and record_num parameters

        Kwargs (ints):
            buffers_per_acquisition
            allocated_buffers
        """
        if 'samples_per_record' in kwargs:
            raise ValueError('With HD_RecordsSamples_Controller '
                             'samples_per_record cannot be set manually '
                             'via update_acquisition_kwargs and should instead'
                             ' be set by setting int_time and int_delay')
        if 'records_per_buffer' in kwargs:
            raise ValueError('With HD_RecordsSamples_Controller '
                             'records_per_buffer cannot be set manually '
                             'via update_acquisition_kwargs and should instead'
                             ' be set by setting record_num')
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
        if self.record_num() != alazar.records_per_buffer.get():
            raise Exception('acq controller records per buffer does not match '
                            'instrument value, most likely need '
                            'to call update_acquisition_kwargs with'
                            'records_per_buffer as a param')

        demod_freqs = self.get_demod_freqs()
        if len(demod_freqs) == 0:
            raise Exception('no demod_freqs set')

        self.buffers_per_acquisition = alazar.buffers_per_acquisition.get()
        self.board_info = alazar.get_idn()
        self.buffer = np.zeros(self.samples_per_record *
                               self.record_num() *
                               self.number_of_channels)

        mat_shape = (self._demod_length, self.record_num(),
                     self.samples_per_record)
        integer_list = np.arange(self.samples_per_record)
        integer_mat = np.outer(np.ones(self.record_num()), integer_list)
        angle_mat = 2 * np.pi * \
            np.outer(demod_freqs, integer_mat).reshape(
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
            - magnitude a numpy array of shape = (demod_length, records_per_buffer)
            - phase a numpy array of shape = (demod_length, records_per_buffer
        """
        # for ATS9360 samples are arranged in the buffer as follows:
        # S00A, S00B, S01A, S01B...S10A, S10B, S11A, S11B...
        # where SXYZ is record X, sample Y, channel Z.

        # break buffer up into records and shapes to be (records, samples)
        reshaped_buf = self.buffer.reshape(self.records_per_buffer,
                                           self.samples_per_record,
                                           self.number_of_channels)
        recordsA = np.uint16(reshaped_buf[:, :, 0] /
                             self.buffers_per_acquisition)

        # TODO(nataliejpg): test above version works on alazar and
        # compare performance
        # recordsA = np.empty((self.record_num(), self.samples_per_record),
        #                     dtype=np.uint16)
        # for i in range(self.record_num()):
        #     i0 = (i * self.samples_per_record * self.number_of_channels)
        #     i1 = (i0 + self.samples_per_record * self.number_of_channels)
        #     recordsA[i, :] = np.uint16(
        #         self.buffer[i0:i1:self.number_of_channels] /
        #         self.buffers_per_acquisitin)

        # do demodulation
        # recordsB = np.uint16(reshaped_buf[:, :, 1] /
        #                      self.buffers_per_acquisition)
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
            # TODO(nataliejpg): think about this recentering makes sense here
            volt_rec = rec - np.mean(rec)

        # volt_rec to matrix and multiply with demodulation signal matrices
        mat_shape = (self._demod_length, self.record_num,
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
                                             axis=-1)
            im_filtered = helpers.filter_win(im_mat, cutoff,
                                             self.sample_rate,
                                             self.filter_settings['numtaps'],
                                             axis=-1)
        elif self.filter_settings['filter'] == 1:
            re_filtered = helpers.filter_ls(re_mat, cutoff,
                                            self.sample_rate,
                                            self.filter_settings['numtaps'],
                                            axis=-1)
            im_filtered = helpers.filter_ls(im_mat, cutoff,
                                            self.sample_rate,
                                            self.filter_settings['numtaps'],
                                            axis=-1)
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
        magnitude = abs(complex_mat)
        phase = np.angle(complex_mat, deg=True)

        return magnitude, phase
