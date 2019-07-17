from qcodes.instrument_drivers.Keysight.keysightb1500 import KeysightB1500, \
    MessageBuilder, constants
from collections import namedtuple

spa = KeysightB1500('spa', address='GPIB21::17::INSTR')
spa.smu1.timing_parameters(0, 0.01, 100)
spa.smu1.measurement_mode(constants.MM.Mode.SAMPLING)
spa.smu1.source_config(constants.VOutputRange.AUTO, 1e-7, None, constants.IOutputRange.AUTO)
spa.smu1.voltage(1e-6)

# spa.smu1.source_config(constants.IOutputRange.AUTO, 10, None, constants.VOutputRange.AUTO)
# spa.smu1.current(1e-6)


spa.write(MessageBuilder().fmt(1,1).message)
raw_data = spa.ask(MessageBuilder().xe().message)


_sampling_index = 'X'
_current_measurement_value = 'I'
_voltage_measurement_value = 'V'
_normal_status = 'N'
_compliance_issue = 'C'
_values_separator = ','
_channel_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

_error_list = {'C': 'Reached_compliance_limit',
               'N': 'Normal',
               'T': 'Another_channel_reached_compliance_limit',
               'V': 'Measured_data_over_measurement_range'
               }


def parse_fmt_1_1_response(resp):
    indices_val = []
    indices_status = []
    indices_channel = []
    indices_datatype = []

    data_val = []
    data_status = []
    data_channel = []
    data_datatype = []

    FMTResponse = namedtuple('FMTResponse', 'value status channel type')

    for str_value in raw_data.split(_values_separator):
        status = str_value[0]
        channel_id = str_value[1]

        datatype = str_value[2]
        value = float(str_value[3:])
        if datatype == _sampling_index:
            indices_val.append(value)
            indices_status.append(status)
            indices_channel.append(channel_id)
            indices_datatype.append(datatype)
        else:
            data_val.append(value)
            data_status.append(status)
            data_channel.append(channel_id)
            data_datatype.append(datatype)

    data = FMTResponse(data_val, data_status, data_channel, data_datatype)
    indices = FMTResponse(indices_val, indices_status, indices_channel, indices_datatype)
    return indices, data


#        _data_type_actions[data_type](value)

#     _data_type_actions = {
#                 _current_measurement_value: data.append,
#                 _voltage_measurement_value: data.append,
#                 _sampling_index: indices.append,
#             }

def compliance_issues(data_status):
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

#     if status == _normal_status:
#         _data_type_actions[data_type](value)
#     elif status in _error_list:
#         raise Exception(f"{_error_list[status]} in response {str_value}")
#     else:
#         raise Exception(f'Not good status in response {str_value}')


