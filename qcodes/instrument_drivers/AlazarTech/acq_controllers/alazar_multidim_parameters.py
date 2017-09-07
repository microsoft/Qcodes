from qcodes import Parameter, MultiParameter, ArrayParameter
import numpy as np

class Alazar0DParameter(Parameter):
    def __init__(self,
                 name,
                 instrument,
                 label,
                 unit,
                 average_buffers=True,
                 average_records=True,
                 integrate_samples=True,
                 shape = (1,),
                 setpoint_names = None,
                 setpoint_labels = None,
                 setpoint_units = None):
        self.acquisition_kwargs = {}
        self._integrate_samples = integrate_samples
        self._average_records = average_records
        self._average_buffers = average_buffers

        super().__init__(name,
                         unit=unit,
                         label=label,
                         instrument=instrument)

    def get(self):
        channel = self._instrument
        cntrl = channel._parent
        params_to_kwargs = ['samples_per_record', 'records_per_buffer',
                            'buffers_per_acquisition', 'allocated_buffers']
        acq_kwargs = self.acquisition_kwargs.copy()
        controller_acq_kwargs = {key: val.get() for key, val in cntrl.parameters.items() if
             key in params_to_kwargs}
        channel_acq_kwargs = {key: val.get() for key, val in channel.parameters.items() if
             key in params_to_kwargs}
        acq_kwargs.update(controller_acq_kwargs)
        acq_kwargs.update(channel_acq_kwargs)

        output = self._instrument._parent._get_alazar().acquire(
            acquisition_controller=self._instrument._parent,
            **acq_kwargs)
        return output

class Alazar1DParameter(ArrayParameter):
    def __init__(self,
                 name,
                 instrument,
                 label,
                 unit,
                 average_buffers=True,
                 average_records=True,
                 integrate_samples=True,
                 shape = (1,),
                 setpoint_names = None,
                 setpoint_labels = None,
                 setpoint_units = None):
        self.acquisition_kwargs = {}
        self._integrate_samples = integrate_samples
        self._average_records = average_records
        self._average_buffers = average_buffers

        if not integrate_samples:
            setpoint_names = ('time',)
            setpoint_labels = ('time',)
            setpoint_units = ('s',)
        if not average_records:
            setpoint_names = ('records',)
            setpoint_labels = ('Records',)
            setpoint_units = ('',)
        if not average_buffers:
            setpoint_names = ('buffers',)
            setpoint_labels = ('Buffers',)
            setpoint_units = ('',)
        super().__init__(name,
                         unit=unit,
                         label=label,
                         shape=shape,
                         instrument=instrument,
                         setpoint_names=setpoint_names,
                         setpoint_labels=setpoint_labels,
                         setpoint_units=setpoint_units)


    def set_setpoints_and_labels(self):
        # int_time = self._instrument.int_time.get() or 0
        # int_delay = self._instrument.int_delay.get() or 0
        # total_time = int_time + int_delay
        if not self._integrate_samples:
            samples = self._instrument._parent.samples_per_record.get()
            sample_rate = self._instrument._parent._get_alazar().get_sample_rate()
            start = 0
            stop = samples/sample_rate
            self.shape = (samples,)
            self.setpoints = (tuple(np.linspace(start, stop, samples)),)
        elif not self._average_records:
            records = self._instrument.records_per_buffer.get()
            start = 0
            stop = records
            self.shape = (records,)
            self.setpoints = (tuple(np.linspace(start, stop, records)),)
        elif not self._average_buffers:
            buffers = self._instrument.buffers_per_acquisition.get()
            start = 0
            stop = buffers
            self.shape = (buffers,)
            self.setpoints = (tuple(np.linspace(start, stop, buffers)),)

    def get(self):
        channel = self._instrument
        cntrl = channel._parent
        params_to_kwargs = ['samples_per_record', 'records_per_buffer',
                            'buffers_per_acquisition', 'allocated_buffers']
        acq_kwargs = self.acquisition_kwargs.copy()
        controller_acq_kwargs = {key: val.get() for key, val in cntrl.parameters.items() if
             key in params_to_kwargs}
        channel_acq_kwargs = {key: val.get() for key, val in channel.parameters.items() if
             key in params_to_kwargs}
        acq_kwargs.update(controller_acq_kwargs)
        acq_kwargs.update(channel_acq_kwargs)

        output = self._instrument._parent._get_alazar().acquire(
            acquisition_controller=self._instrument._parent,
            **acq_kwargs)
        return output

class Alazar2DParameter(ArrayParameter):
    def __init__(self,
                 name,
                 instrument,
                 label,
                 unit,
                 average_buffers=True,
                 average_records=True,
                 integrate_samples=True,
                 shape = (1,1),
                 setpoint_names = None,
                 setpoint_labels = None,
                 setpoint_units = None):
        self.acquisition_kwargs = {}
        self._integrate_samples = integrate_samples
        self._average_records = average_records
        self._average_buffers = average_buffers

        if integrate_samples:
            setpoint_names = ('buffers', 'records')
            setpoint_labels = ('Buffers', 'Records')
            setpoint_units = ('','')
        if average_records:
            setpoint_names = ('buffers', 'time')
            setpoint_labels = ('Buffers', 'Time')
            setpoint_units = ('','S')
        if average_buffers:
            setpoint_names = ('records', 'time')
            setpoint_labels = ('Records', 'Time')
            setpoint_units = ('','S')
        super().__init__(name,
                         unit=unit,
                         label=label,
                         shape=shape,
                         instrument=instrument,
                         setpoint_names=setpoint_names,
                         setpoint_labels=setpoint_labels,
                         setpoint_units=setpoint_units)


    def set_setpoints_and_labels(self):
        records = self._instrument.records_per_buffer()
        buffers = self._instrument.buffers_per_acquisition()
        samples = self._instrument._parent.samples_per_record.get()
        if self._integrate_samples:
            self.shape = (buffers,records)
            inner_setpoints = tuple(np.linspace(0, records, records))
            outer_setpoints = tuple(np.linspace(0, buffers, buffers))
        elif self._average_records:
            sample_rate = self._instrument._parent._get_alazar().get_sample_rate()
            stop = samples/sample_rate
            self.shape = (buffers,samples)
            inner_setpoints = tuple(np.linspace(0, stop, samples))
            outer_setpoints = tuple(np.linspace(0, buffers, buffers))
        elif self._average_buffers:
            sample_rate = self._instrument._parent._get_alazar().get_sample_rate()
            stop = samples/sample_rate
            self.shape = (records,samples)
            inner_setpoints = tuple(np.linspace(0, stop, samples))
            outer_setpoints = tuple(np.linspace(0, records, records))
        else:
            raise RuntimeError("Non supported Array type")
        self.setpoints = (outer_setpoints, tuple(inner_setpoints for _ in range(len(outer_setpoints))))


    def get(self):
        channel = self._instrument
        cntrl = channel._parent
        params_to_kwargs = ['samples_per_record', 'records_per_buffer',
                            'buffers_per_acquisition', 'allocated_buffers']
        acq_kwargs = self.acquisition_kwargs.copy()
        controller_acq_kwargs = {key: val.get() for key, val in cntrl.parameters.items() if
             key in params_to_kwargs}
        channel_acq_kwargs = {key: val.get() for key, val in channel.parameters.items() if
             key in params_to_kwargs}
        acq_kwargs.update(controller_acq_kwargs)
        acq_kwargs.update(channel_acq_kwargs)

        output = self._instrument._parent._get_alazar().acquire(
            acquisition_controller=self._instrument._parent,
            **acq_kwargs)
        return output

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