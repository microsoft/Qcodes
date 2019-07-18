from qcodes.instrument_drivers.Keysight.keysightb1500 import KeysightB1500, \
    MessageBuilder, constants
from collections import namedtuple
from qcodes import ParameterWithSetpoints
import numpy

class SamplingMeasurement(ParameterWithSetpoints):


    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    def _set_up(self):
        self.root_instrument.write(MessageBuilder().fmt(1, 0).message)

    def get_raw(self):
        self._set_up()
        raw_data = self.root_instrument.ask(MessageBuilder().xe().message)
        data = self.parse_fmt_1_0_response(raw_data)
        return numpy.array(data.value)

    def parse_fmt_1_0_response(self,raw_data_val):
        _sampling_index = 'X'
        _current_measurement_value = 'I'
        _voltage_measurement_value = 'V'
        _normal_status = 'N'
        _compliance_issue = 'C'
        _values_separator = ','
        _channel_list = {'A': 'CH1','B': 'CH2','C': 'CH3','D': 'CH4',
                         'E': 'CH5','F': 'CH6','G': 'CH7','H': 'CH8',
                         'I': 'CH9','J': 'CH10','Z': 'XDATA'
                         }

        _error_list = {'C': 'Reached_compliance_limit',
                       'N': 'Normal',
                       'T': 'Another_channel_reached_compliance_limit',
                       'V': 'Measured_data_over_measurement_range'
                       }


        data_val = []
        data_status = []
        data_channel = []
        data_datatype = []

        FMTResponse = namedtuple('FMTResponse', 'value status channel type')

        for str_value in raw_data_val.split(_values_separator):
            status = str_value[0]
            channel_id = _channel_list[str_value[1]]

            datatype = str_value[2]
            value = float(str_value[3:])

            data_val.append(value)
            data_status.append(status)
            data_channel.append(channel_id)
            data_datatype.append(datatype)

        data = FMTResponse(data_val, data_status, data_channel, data_datatype)
        return data

    def compliance_issues(self, data_status):
        """
        check for the status other than "N" (normal) and output the
        number of data values which were not measured under "N" (normal)
        status.
        This includes error such as
        "C" :  compliance limit reached on current channel
        "T" : compliance limit reached on some other channel
        etc
        """

        _total_count = len(data_status)
        _normal_count = data_status.count('N')
        _exception_count = _total_count - _normal_count
        if _total_count == _normal_count:
            print('All measurements are normal')
        else:
            #         raise Exception(f'{str(_exception_count)} measurements were out of compliance')
            indices = [i for i, x in enumerate(data_status) if x == "C" or x == "T"]
            print(f'{str(_exception_count)} measurements were out of compliance at {str(indices)}')






# raw_data = spa.ask(MessageBuilder().xe().message)
# data = parse_fmt_1_0_response(raw_data)
