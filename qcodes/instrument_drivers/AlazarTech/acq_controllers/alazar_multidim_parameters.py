import logging

import numpy as np

from qcodes import Parameter, ArrayParameter
from qcodes.instrument.channel import MultiChannelInstrumentParameter

logger = logging.getLogger(__name__)

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
        alazar_channels = 2
        cntrl.active_channels_nested = [{'ndemods': 0,
                                         'nsignals': 0,
                                         'demod_freqs': [],
                                         'demod_types': [],
                                         'numbers': [],
                                         'raw': False} for _ in range(alazar_channels)]
        alazar_channel = channel.alazar_channel.get()
        channel_info = cntrl.active_channels_nested[alazar_channel]
        channel_info['nsignals'] = 1
        if channel._demod:
            channel_info['ndemods'] = 1
            channel_info['demod_freqs'].append(channel.demod_freq.get())
            channel_info['demod_types'].append(channel.demod_type.get())
        else:
            channel_info['raw'] = True
        cntrl.shape_info['average_buffers'] = channel._average_buffers
        cntrl.shape_info['average_records'] = channel._average_records
        cntrl.shape_info['integrate_samples'] = channel._integrate_samples
        params_to_kwargs = ['samples_per_record', 'records_per_buffer',
                            'buffers_per_acquisition', 'allocated_buffers']
        acq_kwargs = self._instrument.acquisition_kwargs.copy()
        controller_acq_kwargs = {key: val.get() for key, val in cntrl.parameters.items() if
             key in params_to_kwargs}
        channel_acq_kwargs = {key: val.get() for key, val in channel.parameters.items() if
             key in params_to_kwargs}
        acq_kwargs.update(controller_acq_kwargs)
        acq_kwargs.update(channel_acq_kwargs)
        if acq_kwargs['buffers_per_acquisition'] > 1:
            acq_kwargs['allocated_buffers'] = 4
        else:
            acq_kwargs['allocated_buffers'] = 1

        output = self._instrument._parent._get_alazar().acquire(
            acquisition_controller=self._instrument._parent,
            **acq_kwargs)
        logger.info("calling acquire with {}".format(acq_kwargs))
        return output

class AlazarNDParameter(ArrayParameter):
    def __init__(self,
                 name,
                 shape,
                 instrument,
                 label,
                 unit,
                 setpoint_names = None,
                 setpoint_labels = None,
                 setpoint_units = None,
                 average_buffers=True,
                 average_records=True,
                 integrate_samples=True):
        self._integrate_samples = integrate_samples
        self._average_records = average_records
        self._average_buffers = average_buffers
        super().__init__(name,
                         shape=shape,
                         instrument=instrument,
                         label=label,
                         unit=unit,
                         setpoint_names=setpoint_names,
                         setpoint_labels=setpoint_labels,
                         setpoint_units=setpoint_units)

    def get(self):

        channel = self._instrument
        cntrl = channel._parent
        cntrl.shape_info = {}
        alazar_channels = 2
        cntrl.active_channels_nested = [{'ndemods': 0,
                                         'nsignals': 0,
                                         'demod_freqs': [],
                                         'demod_types': [],
                                         'numbers': [],
                                         'raw': False} for _ in range(alazar_channels)]
        alazar_channel = channel.alazar_channel.get()
        channel_info = cntrl.active_channels_nested[alazar_channel]
        channel_info['nsignals'] = 1
        if channel._demod:
            channel_info['ndemods'] = 1
            channel_info['demod_freqs'].append(channel.demod_freq.get())
            channel_info['demod_types'].append(channel.demod_type.get())
        else:
            channel_info['raw'] = True
        cntrl.shape_info['average_buffers'] = channel._average_buffers
        cntrl.shape_info['average_records'] = channel._average_records
        cntrl.shape_info['integrate_samples'] = channel._integrate_samples

        params_to_kwargs = ['samples_per_record', 'records_per_buffer',
                            'buffers_per_acquisition', 'allocated_buffers']
        acq_kwargs = self._instrument.acquisition_kwargs.copy()
        controller_acq_kwargs = {key: val.get() for key, val in cntrl.parameters.items() if
             key in params_to_kwargs}
        channel_acq_kwargs = {key: val.get() for key, val in channel.parameters.items() if
             key in params_to_kwargs}
        acq_kwargs.update(controller_acq_kwargs)
        acq_kwargs.update(channel_acq_kwargs)
        if acq_kwargs['buffers_per_acquisition'] > 1:
            acq_kwargs['allocated_buffers'] = 4
        else:
            acq_kwargs['allocated_buffers'] = 1

        logger.info("calling acquire with {}".format(acq_kwargs))
        output = self._instrument._parent._get_alazar().acquire(
            acquisition_controller=self._instrument._parent,
            **acq_kwargs)
        return output

class Alazar1DParameter(AlazarNDParameter):
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
                         instrument=instrument,
                         label=label,
                         shape=shape,
                         setpoint_names=setpoint_names,
                         setpoint_labels=setpoint_labels,
                         setpoint_units=setpoint_units,
                         average_buffers=average_buffers,
                         average_records=average_records,
                         integrate_samples=integrate_samples)

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

class Alazar2DParameter(AlazarNDParameter):
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
                         setpoint_units=setpoint_units,
                         average_buffers=average_buffers,
                         average_records=average_records,
                         integrate_samples=integrate_samples)

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


class AlazarMultiChannelParameter(MultiChannelInstrumentParameter):
    """


    """
    def get(self):
        if self._param_name == 'data':
            channel = self._channels[0]
            cntrl = channel._parent
            instrument = cntrl._get_alazar()
            cntrl.shape_info = {}
            alazar_channels = 2
            cntrl.active_channels_nested = [{'ndemods':0,
                                             'nsignals':0,
                                             'demod_freqs':[],
                                             'demod_types': [],
                                             'numbers': [],
                                             'raw':False} for _ in range(alazar_channels)]

            for i, channel in enumerate(self._channels):
                # change this to use raw value once mapping is
                # complete
                alazar_channel = channel.alazar_channel.get()
                channel_info = cntrl.active_channels_nested[alazar_channel]
                channel_info['nsignals'] += 1

                if channel._demod:
                    channel_info['ndemods'] += 1
                    channel_info['demod_freqs'].append(channel.demod_freq.get())
                    channel_info['demod_types'].append(channel.demod_type.get())
                else:
                    channel_info['raw'] = True
                cntrl.shape_info['average_buffers'] = channel._average_buffers
                cntrl.shape_info['average_records'] = channel._average_records
                cntrl.shape_info['integrate_samples'] = channel._integrate_samples
                cntrl.shape_info['channel'] = channel.alazar_channel.get()

            params_to_kwargs = ['samples_per_record', 'records_per_buffer',
                                'buffers_per_acquisition', 'allocated_buffers']
            acq_kwargs = channel.acquisition_kwargs.copy()
            controller_acq_kwargs = {key: val.get() for key, val in cntrl.parameters.items() if
                 key in params_to_kwargs}
            channels_acq_kwargs = []
            for i, channel in enumerate(self._channels):
                channels_acq_kwargs.append({key: val.get() for key, val in channel.parameters.items() if
                     key in params_to_kwargs})
                if channels_acq_kwargs[i] != channels_acq_kwargs[0]:
                    raise RuntimeError("kwargs are not the same")
            acq_kwargs.update(controller_acq_kwargs)
            acq_kwargs.update(channels_acq_kwargs[0])
            if acq_kwargs['buffers_per_acquisition'] > 1:
                acq_kwargs['allocated_buffers'] = 4
            else:
                acq_kwargs['allocated_buffers'] = 1

            logger.info("calling acquire with {}".format(acq_kwargs))
            output = instrument.acquire(
                acquisition_controller=cntrl,
                **acq_kwargs)
        else:
            output = tuple(chan.parameters[self._param_name].get()
                           for chan in self._channels)
        return output
