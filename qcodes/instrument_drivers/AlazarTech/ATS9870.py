from .ATS import AlazarTech_ATS, AlazarParameter
from qcodes.utils import validators

class AlazarTech_ATS9870(AlazarTech_ATS):
    def __init__(self, name):
        super().__init__(name)
        # add parameters

        self.add_parameter(name='clock_source', parameter_class=AlazarParameter, label='Clock Source', unit=None,
                           value='INTERNAL_CLOCK',
                           byte_to_value_dict={1: 'INTERNAL_CLOCK',
                                               4: 'SLOW_EXTERNAL_CLOCK',
                                               5: 'EXTERNAL_CLOCK_AC',
                                               7: 'EXTERNAL_CLOCK_10_MHz_REF'})
        self.add_parameter(name='sample_rate', parameter_class=AlazarParameter, label='Sample Rate', unit='S/s',
                           value=1000000000,
                           byte_to_value_dict={0x1: 1000, 0x2: 2000, 0x4: 5000,
                                               0x8: 10000, 0xA: 20000, 0xC: 50000,
                                               0xE: 100000, 0x10: 200000, 0x12: 500000,
                                               0x14: 1000000, 0x18: 2000000, 0x1A: 5000000,
                                               0x1C: 10000000, 0x1E: 20000000, 0x22: 50000000,
                                               0x24: 100000000, 0x2B: 250000000, 0x30: 500000000,
                                               0x35: 1000000000,
                                               0x40: 'EXTERNAL_CLOCK',
                                               1000000000: '1GHz_REFERENCE_CLOCK'})
        self.add_parameter(name='clock_edge', parameter_class=AlazarParameter, label='Clock Edge', unit=None,
                           value='CLOCK_EDGE_RISING',
                           byte_to_value_dict={0: 'CLOCK_EDGE_RISING',
                                               1: 'CLOCK_EDGE_FALLING'})

        self.add_parameter(name='decimation', parameter_class=AlazarParameter, label='Decimation', unit=None,
                           value=0, vals=validators.Ints(0, 100000))

        # TODO (M) make parameter for board type

        # TODO (M) check board kind

# -----config-----
#         sample_rate
#         clock_edge
#         decimation
#
# coupling{n}
# range{n}
# impedance{n}
# bwlimit{n}
#
# trigger_operation
# trigger_engine1
# trigger_source1
# trigger_slope1
# trigger_level1
# trigger_engine2
# trigger_source2
# trigger_slope2
# trigger_level2
#
# external_trigger_coupling
# trigger_range
#
# trigger_delay
# timeout_ticks
#
# ----acquire-----
# mode
# samples_per_record
# channel_selection
#
# more !?
#
# nbuffers


