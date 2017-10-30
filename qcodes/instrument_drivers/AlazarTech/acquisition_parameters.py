from qcodes import Parameter, MultiParameter, ArrayParameter
import numpy as np
import logging

class AcqVariablesParam(Parameter):
    """
    Parameter of an AcquisitionController which has a _check_and_update_instr
    function used for validation and to update instrument attributes and a
    _get_default function which it uses to set the AcqVariablesParam to an
    instrument calculated default.

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
        self._check_and_update_instr = check_and_update_fn
        if default_fn is not None:
            self._get_default = default_fn

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
        return self._latest['value']

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
        val = self._latest['value']
        self._check_and_update_instr(val, param_name=self.name)
        return True


class AlazarMultiArray(MultiParameter):
    """
    Hardware controlled parameter class for Alazar acquisition. To be used with
    Acquisition Controller (tested with ATS9360 board)

    Alazar Instrument 'acquire' returns a buffer of data each time a buffer is
    filled (channels * samples * records) which is processed by the
    post_acquire function of the Acquisition Controller and finally the
    processed result is returned when the AlazarMultiArray parameter is called.

    Args:
        name: name for this parameter
        names: names of the two arrays returned from get
        instrument: acquisition controller instrument this parameter belongs to
        demod_length (int): number of demodulators. Default 1
    """

    def __init__(self, name, instrument, names=("A", "B"), demod_length=1):
        if demod_length > 1:
            shapes = ((demod_length, ), (demod_length, ))
        else:
            shapes = ((), ())
        super().__init__(name, names=names, shapes=shapes, instrument=instrument)
        self.acquisition_kwargs = {}

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
            # self.setpoints = ((demod_freqs, ), (demod_freqs, ))
        else:
            pass

    def get(self):
        """
        Gets the output by calling acquire on the alazar (which in turn calls the
        processing functions of the acquisition controller before returning the reshaped data.
        The exact data will depend on the AcquisitionController used

        returns:
            - A a numpy array of Alazar data
            - B a numpy array of Alazar data
        """
        a, b = self._instrument._get_alazar().acquire(
            acquisition_controller=self._instrument,
            **self.acquisition_kwargs)
        return a, b


class AlazarMultiArray2D(AlazarMultiArray):

    def update_sweep(self, start, stop, npts):
        """
        Function which updates the shape of the parameter (and it's setpoints
        when this is fixed)

        Args:
            start: start time of samples returned after processing
            stop: stop time of samples returned after processing
            npts: number of samples returned after processing
        """
        demod_length = self._instrument._demod_length
        # self._time_list = tuple(np.linspace(start, stop, num=npts)
        if demod_length > 1:
            # demod_freqs = self._instrument.get_demod_freqs()
            # self.setpoints = ((demod_freqs, self._time_list),
            #                   (demod_freqs, self._time_list))
            self.shapes = ((demod_length, npts), (demod_length, npts))
        else:
            self.shapes = ((npts,), (npts,))
            # self.setpoints = ((self._time_list,), (self._time_list,))


class AlazarMultiArray3D(AlazarMultiArray):

    def update_buf_sweep(self, buf_npts, buf_start=None, buf_stop=None):
        """
        Function which updates the shape of the parameter (and it's setpoints
        when this is fixed)

        Args:
            buf_npts: number of buffers returned
            buf_start (optional): start value of buffers returned
            buf_stop (optional): stop value of records returned
        """
        demod_length = self._instrument._demod_length
        # self._buf_list = tuple(np.linspace(buf_start,
        #                                    buf_stop, num=buf_npts))
        self._buf_npts = buf_npts
        if demod_length > 1:
            # demod_freqs = self._instrument.get_demod_freqs()
            # self.setpoints = ((demod_freqs, self._buf_list, self._rec_list),
            #                   (demod_freqs, self._buf_list, self._rec_list))
            self.shapes = ((demod_length, self._buf_npts, self._rec_npts),
                           (demod_length, self._buf_npts, self._rec_npts))
        else:
            self.shapes = ((self._buf_npts, self._rec_npts),
                           (self._buf_npts, self._rec_npts))
            # self.setpoints = ((self._buf_list, self._rec_list),
            #                   (self._buf_list, self._rec_list))

    def update_rec_sweep(self, rec_npts, rec_start=None, rec_stop=None):
        """
        Function which updates the shape of the parameter (and it's setpoints
        when this is fixed)

        Args:
            rec_npts: number of records returned after processing
            rec_start (optional): start value of records returned
            rec_stop (optional): stop value of records returned
        """
        demod_length = self._instrument._demod_length
        # self._rec_list = tuple(np.linspace(rec_start,
        #                                    rec_stop, num=rec_npts))
        self._rec_npts = rec_npts
        if demod_length > 1:
            # demod_freqs = self._instrument.get_demod_freqs()
            # self.setpoints = ((demod_freqs, self._buf_list, self._rec_list),
            #                   (demod_freqs, self._buf_list, self._rec_list))
            self.shapes = ((demod_length, self._buf_npts, self._rec_npts),
                           (demod_length, self._buf_npts, self._rec_npts))
        else:
            self.shapes = ((self._buf_npts, self._rec_npts),
                           (self._buf_npts, self._rec_npts))
            # self.setpoints = ((self._buf_list, self._rec_list),
            #                   (self._buf_list, self._rec_list))


class ExpandingAlazarArrayMultiParameter(MultiParameter):
    def __init__(self,
                 name,
                 instrument,
                 names = ('raw_output',),
                 labels = ("raw output",),
                 units = ('v',),
                 shapes = ((1,),),
                 setpoints = (((1,),),),
                 setpoint_names = None,
                 setpoint_labels = None,
                 setpoint_units = None,
                 integrate_samples=False,
                 average_records=True):
        self.acquisition_kwargs = {}
        self._integrate_samples = integrate_samples
        self._average_records = average_records

        if setpoint_names:
            self.setpoint_names_base = setpoint_names[0]
        elif integrate_samples and average_records:
            self.setpoint_names_base = ()
        elif integrate_samples:
            self.setpoint_names_base = ('record_num',)
        elif average_records:
            self.setpoint_names_base = ('time',)
        if setpoint_labels:
            self.setpoint_labels_base = setpoint_names[0]
        elif integrate_samples and average_records:
            self.setpoint_labels_base = ()
        elif integrate_samples:
            self.setpoint_labels_base = ('record num',)
        elif average_records:
            self.setpoint_labels_base = ('time',)
        if setpoint_units:
            self.setpoint_units_base = setpoint_units[0]
        elif integrate_samples and average_records:
            self.setpoint_units_base = ()
        elif integrate_samples:
            self.setpoint_units_base = ('',)
        elif average_records:
            self.setpoint_units_base = ('s',)

        self.setpoints_start = 0
        self.setpoints_stop = 0

        super().__init__(name,
                         names=names,
                         units=units,
                         labels=labels,
                         shapes=shapes,
                         instrument=instrument,
                         setpoints=setpoints,
                         setpoint_names=setpoint_names,
                         setpoint_labels=setpoint_labels,
                         setpoint_units=setpoint_units)



    def set_base_setpoints(self, base_name=None, base_label=None, base_unit=None,
                           setpoints_start=None, setpoints_stop=None):
        if base_name is not None:
            self.setpoint_names_base = (base_name,)
        if base_label is not None:
            self.setpoint_labels_base = (base_label,)
        if base_unit is not None:
            self.setpoint_units_base = (base_unit,)
        if setpoints_start is not None:
            self.setpoints_start = setpoints_start
        if setpoints_stop is not None:
            self.setpoints_stop = setpoints_stop
        self.set_setpoints_and_labels()

    def set_setpoints_and_labels(self):
        if not self._integrate_samples:
            int_time = self._instrument.int_time.get() or 0
            int_delay = self._instrument.int_delay.get() or 0
            total_time = int_time + int_delay
            samples = self._instrument.samples_per_record.get()
            if total_time and samples:
                start = 0
                stop = total_time
            else:
                start = 0
                stop = 1
            samples = samples or 1
            arraysetpoints = (tuple(np.linspace(start, stop, samples)),)
            base_shape = (len(arraysetpoints[0]),)
        elif not self._average_records:
            num_records = self._instrument.records_per_buffer.get() or 0
            start = self.setpoints_start
            stop = self.setpoints_stop or num_records-1
            arraysetpoints = (tuple(np.linspace(start, stop, num_records)),)
            base_shape = (self._instrument.records_per_buffer.get(),)
        else:
            arraysetpoints = ()
            base_shape = ()
        setpoints = [arraysetpoints]
        names = [self.names[0]]
        labels = [self.labels[0]]
        setpoint_names = [self.setpoint_names_base]
        setpoint_labels = [self.setpoint_labels_base]
        setpoint_units = [self.setpoint_units_base]
        units = [self.units[0]]
        shapes = [base_shape]
        demod_freqs = self._instrument.demod_freqs.get()
        for i, demod_freq in enumerate(demod_freqs):
            names.append("demod_freq_{}_mag".format(i))
            labels.append("demod freq {} mag".format(i))
            names.append("demod_freq_{}_phase".format(i))
            labels.append("demod freq {} phase".format(i))
            units.append('v')
            units.append('v')
            shapes.append(base_shape)
            shapes.append(base_shape)
            setpoints.append(arraysetpoints)
            setpoint_names.append(self.setpoint_names_base)
            setpoint_labels.append(self.setpoint_labels_base)
            setpoint_units.append(self.setpoint_units_base)
            setpoints.append(arraysetpoints)
            setpoint_names.append(self.setpoint_names_base)
            setpoint_labels.append(self.setpoint_labels_base)
            setpoint_units.append(self.setpoint_units_base)
        self.names = tuple(names)
        self.labels = tuple(labels)
        self.units = tuple(units)
        self.shapes = tuple(shapes)
        self.setpoints = tuple(setpoints)
        self.setpoint_names = tuple(setpoint_names)
        self.setpoint_labels = tuple(setpoint_labels)
        self.setpoint_units = tuple(setpoint_units)

    def get(self):
        inst = self._instrument
        params_to_kwargs = ['samples_per_record', 'records_per_buffer',
                            'buffers_per_acquisition', 'allocated_buffers']
        acq_kwargs = self.acquisition_kwargs.copy()
        additional_acq_kwargs = {key: val.get() for key, val in inst.parameters.items() if
             key in params_to_kwargs}
        acq_kwargs.update(additional_acq_kwargs)

        output = self._instrument._get_alazar().acquire(
            acquisition_controller=self._instrument,
            **acq_kwargs)
        return output


class NonSettableDerivedParameter(Parameter):
    """
    Parameter of an AcquisitionController which cannot be updated directly
    as it's value is derived from other parameters. This is intended to be
    used in high level APIs where Alazar parameters such as 'samples_per_record'
    are not set directly but are parameters of the actual instrument anyway.

    This assumes that the parameter is stored via a call to '_save_val' by
    any set of parameter that this parameter depends on.

    Args:
        name: name for this parameter
        instrument: acquisition controller instrument this parameter belongs to
        alternative (str): name of parameter(s) that controls the value of this
            parameter and can be set directly.
    """

    def __init__(self, name, instrument, alternative: str, **kwargs):
        self._alternative = alternative
        super().__init__(name, instrument=instrument, **kwargs)

    def set(self, value):
        """
        It's not possible to directly set this parameter as it's derived from other
        parameters.
        """
        raise NotImplementedError("Cannot directly set {}. To control this parameter"
                                  "set {}".format(self.name, self._alternative))

    def get(self):
        return self.get_latest()


class EffectiveSampleRateParameter(NonSettableDerivedParameter):


    def get(self):
        """
        Obtain the effective sampling rate of the acquisition
        based on clock type, clock speed and decimation

        Returns:
            the number of samples (per channel) per second
        """
        if self._instrument.clock_source.get() == 'EXTERNAL_CLOCK_10MHz_REF':
            rate = self._instrument.external_sample_rate.get()
        elif self._instrument.clock_source.get() == 'INTERNAL_CLOCK':
            rate = self._instrument.sample_rate.get()
        else:
            raise Exception("Don't know how to get sample rate with {}".format(self._instrument.clock_source.get()))

        if rate == '1GHz_REFERENCE_CLOCK':
            rate = 1e9

        decimation = self._instrument.decimation.get()
        if decimation > 0:
            rate = rate / decimation

        self._save_val(rate)
        return rate


class DemodFreqParameter(ArrayParameter):

    #
    def __init__(self, name, shape, **kwargs):
        self._demod_freqs = []
        super().__init__(name, shape, **kwargs)


    def add_demodulator(self, demod_freq):
        ndemod_freqs = len(self._demod_freqs)
        if demod_freq not in self._demod_freqs:
            self._verify_demod_freq(demod_freq)
            self._demod_freqs.append(demod_freq)
            self._save_val(self._demod_freqs)
            self.shape = (ndemod_freqs+1,)
            self._instrument.acquisition.set_setpoints_and_labels()

    def remove_demodulator(self, demod_freq):
        ndemod_freqs = len(self._demod_freqs)
        if demod_freq in self._demod_freqs:
            self._demod_freqs.pop(self._demod_freqs.index(demod_freq))
            self.shape = (ndemod_freqs - 1,)
            self._save_val(self._demod_freqs)
            self._instrument.acquisition.set_setpoints_and_labels()

    def get(self):
        return self._demod_freqs

    def get_num_demods(self):
        return len(self._demod_freqs)

    def get_max_demod_freq(self):
        if len(self._demod_freqs):
            return max(self._demod_freqs)
        else:
            return None

    def _verify_demod_freq(self, value):
        """
        Function to validate a demodulation frequency

        Checks:
            - 1e6 <= value <= 500e6
            - number of oscillation measured using current 'int_time' param value
              at this demodulation frequency value
            - oversampling rate for this demodulation frequency

        Args:
            value: proposed demodulation frequency
        Returns:
            bool: Returns True if suitable number of oscillations are measured and
            oversampling is > 1, False otherwise.
        Raises:
            ValueError: If value is not a valid demodulation frequency.
        """
        if (value is None) or not (1e6 <= value <= 500e6):
            raise ValueError('demod_freqs must be 1e6 <= value <= 500e6')
        isValid = True
        alazar = self._instrument._get_alazar()
        sample_rate = alazar.effective_sample_rate.get()
        int_time = self._instrument.int_time.get()
        min_oscillations_measured = int_time * value
        oversampling = sample_rate / (2 * value)
        if min_oscillations_measured < 10:
            isValid = False
            logging.warning('{} oscillation measured for largest '
                            'demod freq, recommend at least 10: '
                            'decrease sampling rate, take '
                            'more samples or increase demodulation '
                            'freq'.format(min_oscillations_measured))
        elif oversampling < 1:
            isValid = False
            logging.warning('oversampling rate is {}, recommend > 1: '
                            'increase sampling rate or decrease '
                            'demodulation frequency'.format(oversampling))

        return isValid