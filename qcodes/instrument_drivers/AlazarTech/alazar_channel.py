from qcodes.instrument.channel import InstrumentChannel
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils import validators as vals

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


    def __init__(self, parent, name: str, demod: bool=True, alazar_channel: str='A'):


        super().__init__(parent, name)



        if demod:
            self.add_parameter('demod_freq',
                               label='demod freq',
                               vals=vals.Numbers(1e6,500e6),
                               parameter_class=ManualParameter)

        self.add_parameter('alazar_channel',
                           label='Alazar Channel',
                           parameter_class=ManualParameter)

        # self.add_parameter('single_point',
        #                    label='single point')
        #
        # self.add_parameter('record_trace')
        #
        # self.add_parameter('buffers_vs_records_trace')
        #
        # self.add_parameter('samples_trace')
        #
        # self.add_parameter('records_vs_samples_trace')


