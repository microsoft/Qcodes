from qcodes.instrument.channel import InstrumentChannel
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals
from .alazar_multidim_parameters import Alazar0DParameter, Alazar1DParameter, Alazar2DParameter

class AlazarChannel(InstrumentChannel):
    """
    A single channel for Alazar card. This can capture and return multiple different views of the data

    An Alazar acquisition consists of one or more buffers, each buffer contains on or more records and each
    records contains a number of samples. The timeseries (samples as a function of time) may optionally be demodulated
    by a user selected frequency.

    single_point: Averaged over Buffers and Records and integrated over samples
    records_trace: Averaged over buffers and integrated over samples. 1D trace as a function of records.
    buffers_vs_records_trace: Integrated over samples. 2D array of buffers vs records
    samples_trace: Averaged over buffers and records. 1D trace as a function of samples (time)
    records_vs_samples_trace: Averaged over buffers. 2D array of records vs samples

    """


    def __init__(self, parent, name: str, demod: bool=True, alazar_channel: str='A',
                 average_buffers: bool=True, average_records=True, integrate_samples=True):


        super().__init__(parent, name)

        self.dimensions = 3 - int(average_buffers) - int(average_records) - int(integrate_samples)

        self._average_buffers = average_buffers
        self._average_records = average_records
        self._integrate_samples = integrate_samples

        if self.dimensions >= 3:
            raise RuntimeError("Alazar controller only supports up to 2 dimensional arrays")

        self._demod = demod
        if demod:
            self.add_parameter('demod_freq',
                               label='demod freq',
                               vals=vals.Numbers(1e6,500e6),
                               parameter_class=ManualParameter)

        self.add_parameter('alazar_channel',
                           label='Alazar Channel',
                           parameter_class=ManualParameter)
        self.add_parameter('samples_per_record',
                           parameter_class=ManualParameter)
        self.add_parameter('records_per_buffer',
                           label='records_per_buffer',
                           parameter_class=ManualParameter)
        self.add_parameter('buffers_per_acquisition',
                           label='records_per_buffer',
                           parameter_class=ManualParameter)
        if self.dimensions == 0:
            self.add_parameter('data',
                               label='mydata',
                               unit='V',
                               integrate_samples=integrate_samples,
                               average_records=average_records,
                               average_buffers=average_buffers,
                               parameter_class=Alazar0DParameter)
        elif self.dimensions == 1:
            self.add_parameter('data',
                               label='mydata',
                               unit='V',
                               integrate_samples=integrate_samples,
                               average_records=average_records,
                               average_buffers=average_buffers,
                               parameter_class=Alazar1DParameter)
        elif self.dimensions == 2:
            self.add_parameter('data',
                               label='mydata',
                               unit='V',
                               integrate_samples=integrate_samples,
                               average_records=average_records,
                               average_buffers=average_buffers,
                               parameter_class=Alazar2DParameter)
        else:
            raise RuntimeError("Not implemented here")

        self.acquisition_kwargs = {}

    def prepare_channel(self):
        self._parent.active_channels.append({})
        self._parent.active_channels[0]['demod'] = self._demod
        if self._demod:
            self._parent.active_channels[0]['demod_freq'] = self.demod_freq.get()
        else:
            self._parent.active_channels[0]['demod_freq'] = None
        self._parent.active_channels[0]['average_buffers'] = self._average_buffers
        self._parent.active_channels[0]['average_records'] = self._average_records
        self._parent.active_channels[0]['integrate_samples'] = self._integrate_samples
        self._parent.active_channels[0]['channel'] = self.alazar_channel.get()
        if self.dimensions > 0:
            self.data.set_setpoints_and_labels()
