import math
from qcodes.instrument.channel import InstrumentChannel
from qcodes.utils import validators as vals
from .alazar_multidim_parameters import Alazar0DParameter, Alazar1DParameter, Alazar2DParameter
from ..acquisition_parameters import AcqVariablesParam, NonSettableDerivedParameter

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
                               vals=vals.Numbers(1e5,500e6),
                               get_cmd=None, set_cmd=None)
            self.add_parameter('demod_type',
                               label='demod type',
                               vals=vals.Strings(),
                               get_cmd=None, set_cmd=None)

        self.add_parameter('alazar_channel',
                           label='Alazar Channel',
                           # once val mapping is fixed for
                           # parameters
                           #val_mapping={'A': 0, 'B': 1},
                           get_cmd=None, set_cmd=None)
        if not average_records:
            self.add_parameter('records_per_buffer',
                               label='records_per_buffer',
                               get_cmd=None, set_cmd=None)
        else:
            self.add_parameter('records_per_buffer',
                               label='records_per_buffer',
                               alternative='num_averages',
                               parameter_class=NonSettableDerivedParameter)
        if not average_buffers:
            self.add_parameter('buffers_per_acquisition',
                               label='records_per_buffer',
                               get_cmd=None, set_cmd=None)
        else:
            self.add_parameter('buffers_per_acquisition',
                               label='records_per_buffer',
                               alternative='num_averages',
                               parameter_class=NonSettableDerivedParameter)
        self.add_parameter('num_averages',
                           #label='num averages',
                           check_and_update_fn=self._update_num_avg,
                           default_fn= lambda : 1,
                           parameter_class=AcqVariablesParam)
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
        if self.dimensions > 0:
            self.data.set_setpoints_and_labels()

    def _update_num_avg(self, value: int, **kwargs):
        if not self._average_buffers and not self._average_records:
            if value==1:
                return
            else:
                raise RuntimeError("You requested averaging but are neither averaging over buffers or records")
        if self._average_buffers and not self._average_records:
            self.buffers_per_acquisition._save_val(value)
        elif self._average_records and not self._average_buffers:
            self.records_per_buffer._save_val(value)
        elif self._average_buffers and self._average_records:
            max_samples = self._parent.board_info['max_samples']
            samples_per_rec = self._parent.samples_per_record()
            tot_samples = value * samples_per_rec
            if tot_samples > max_samples:
                records = math.floor(max_samples/samples_per_rec)
                buffers = math.ceil(max_samples/records)
            else:
                records = value
                buffers = 1
            self.buffers_per_acquisition._save_val(buffers)
            self.records_per_buffer._save_val(records)
