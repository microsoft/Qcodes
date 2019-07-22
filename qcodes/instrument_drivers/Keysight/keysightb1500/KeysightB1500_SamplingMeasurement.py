from qcodes.instrument_drivers.Keysight.keysightb1500 import KeysightB1500, \
    MessageBuilder, constants
from collections import namedtuple
from qcodes import ParameterWithSetpoints
import numpy
"""
Initialize data in the begining 
"""

class SamplingMeasurement(ParameterWithSetpoints):
    """
    Performs sampling measurement using semiconductor
    parameter analyzer B1500A.
    """

    _timeout_response_factor  = 5
    # This factor is a bit higher than the ratio between
    # the measured measurement-time and the calculated measurement
    # (from the user input). Check :get_raw: method to find its usage.

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.data = []

    def _set_up(self):
        self.root_instrument.write(MessageBuilder().fmt(1, 0).message)

    def get_raw(self):
        """
        sets up the measurement by calling :_set_up:

        Automatically sets up the visa time out.

        The visa time-out should be longer than the time it takes to finish
        the sampling measurement. The reason is that while measurement is running
        the data keeps on appending in the buffer of SPA. Only when the measurement
        is finished the data is returned to the VISA handle.
        Hence during this time the VISA is idle and waiting for the response.
        If the timeout is lower than the total run time of the measurement,
        VISA will give error.

        We set the Visa timeout to be the measurement_time times the _timeout_response_factor  .
        Strictly speaking the timeout should be just higher the measurement time.

        :return: numpy array with sampling measurement
        """

        measurement_time = self.instrument._total_measurement_time()
        with self.root_instrument.timeout.set_to(measurement_time * self._timeout_response_factor):
            self._set_up()
            raw_data = self.root_instrument.ask(MessageBuilder().xe().message)
            self.data = self.parse_fmt_1_0_response(raw_data)
        return numpy.array(self.data.value)

    def parse_fmt_1_0_response(self,raw_data_val):
        """
        parse the raw data from SPA into a named tuple
        with names
            value: data
            status: Normal or with compliance error
            such as "C","T","V"
            channel: channel number of the output data
            such as "CH1","CH2" ..
            type: current "I" or voltage "V"

        :Args
            raw_data_val: Unparsed (raw) data for the instrument
        """

        values_separator = ','
        """
        PUT THIS OUTSIDE THIS FUNCTION AS A CONSTANT
        """
        channel_list = {'A': 'CH1','B': 'CH2','C': 'CH3','D': 'CH4',
                         'E': 'CH5','F': 'CH6','G': 'CH7','H': 'CH8',
                         'I': 'CH9','J': 'CH10','Z': 'XDATA'
                         }

        """
        DEFINE THIS AS AN ENUM
        """

        _error_list = {'C': 'Reached_compliance_limit',
                       'N': 'Normal',
                       'T': 'Another_channel_reached_compliance_limit',
                       'V': 'Measured_data_over_measurement_range'
                       }

        """
        OUTPUT AS NUMPY ARRAY
        """
        data_val = []
        data_status = []
        data_channel = []
        data_datatype = []

        FMTResponse = namedtuple('FMTResponse', 'value status channel type')

        for str_value in raw_data_val.split(values_separator):
            status = str_value[0]
            channel_id = channel_list[str_value[1]]

            datatype = str_value[2]
            value = float(str_value[3:])

            data_val.append(value)
            data_status.append(status)
            data_channel.append(channel_id)
            data_datatype.append(datatype)

        data = FMTResponse(data_val, data_status, data_channel, data_datatype)
        return data

    def compliance(self):
        """
        check for the status other than "N" (normal) and output the
        number of data values which were not measured under "N" (normal)
        status.

        For the list of all the status values and their meaning refer to :class:`constants.Statuses`.

        This includes error such as
        "C" :  compliance limit reached on current channel
        "T" : compliance limit reached on some other channel
        etc
        """
        if self.data is not None:
            data = self.data
            error_list = {'C': 0,'N': 1,'T': 0,'V': 0}
            total_count = len(data.status)
            normal_count = data.status.count('N')
            exception_count = total_count - normal_count
            if total_count == normal_count:
                print('All measurements are normal')
            else:
                indices = [i for i, x in enumerate(data.status) if x == "C" or x == "T"]
                print(f'{str(_exception_count)} measurements were out of compliance at {str(indices)}')

            compliance_list = [error_list[key] for key in data.status]
            return compliance_list
        else:
            print('First run "sampling_measurement.get()" method to generate the data')
