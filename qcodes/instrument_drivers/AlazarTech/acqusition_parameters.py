from qcodes import Parameter, MultiParameter
import numpy as np

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
                 units = ('V',),
                 shapes = ((1,),),
                 setpoints = (((1,),),),
                 setpoint_names = (('time',),),
                 setpoint_labels = (('time',),),
                 setpoint_units = (('s',),),
                 integrate_samples=False):
        self.acquisition_kwargs = {}
        self._integrate_samples = integrate_samples
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

    def set_setpoints_and_labels(self):
        if not self._integrate_samples:

            if self._instrument._get_alazar().sample_rate.get() and self.acquisition_kwargs.get('samples_per_record'):
                start = 0
                samples = self.acquisition_kwargs['samples_per_record']
                stop = samples/self._instrument._get_alazar().sample_rate.get()
            else:
                start = 0
                samples = 1
                stop = 1
            print("start {} stop {} num steps {}".format(start, stop, samples))
            arraysetpoints = (tuple(np.linspace(start, stop, samples)),)
            base_shape = (len(arraysetpoints[0]),)
        else:
            arraysetpoints = ()
            base_shape = ()
        setpoints = [arraysetpoints]
        names = [self.names[0]]
        labels = [self.labels[0]]
        setpoint_name = self.setpoint_names[0]
        setpoint_names = [setpoint_name]
        setpoint_label = self.setpoint_labels[0]
        setpoint_labels = [setpoint_label]
        setpoint_unit = self.setpoint_units[0]
        setpoint_units = [setpoint_unit]
        units = [self.units[0]]
        shapes = [base_shape]
        for i, demod_freq in enumerate(self._instrument._demod_freqs):
            names.append("demod_freq_{}_mag".format(i))
            labels.append("demod freq {} mag".format(i))
            names.append("demod_freq_{}_phase".format(i))
            labels.append("demod freq {} phase".format(i))
            units.append('v')
            units.append('v')
            shapes.append(base_shape)
            shapes.append(base_shape)
            setpoints.append(arraysetpoints)
            setpoint_names.append(setpoint_name)
            setpoint_labels.append(setpoint_label)
            setpoint_units.append(setpoint_unit)
            setpoints.append(arraysetpoints)
            setpoint_names.append(setpoint_name)
            setpoint_labels.append(setpoint_label)
            setpoint_units.append(setpoint_unit)
        self.metadata['demod_freqs'] = self._instrument._demod_freqs
        self.names = tuple(names)
        self.labels = tuple(labels)
        self.units = tuple(units)
        self.shapes = tuple(shapes)
        self.setpoints = tuple(setpoints)
        self.setpoint_names = tuple(setpoint_names)
        self.setpoint_labels = tuple(setpoint_labels)
        self.setpoint_units = tuple(setpoint_units)

    def get(self):
        output = self._instrument._get_alazar().acquire(
            acquisition_controller=self._instrument,
            **self.acquisition_kwargs)
        return output