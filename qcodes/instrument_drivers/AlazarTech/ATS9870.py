from .ATS import AlazarTech_ATS, AlazarParameter


class AlazarTech_ATS9870(AlazarTech_ATS):
    def __init__(self, name):
        super().__init__(name)
        # add parameters

        self.add_parameter(name='clock_source', parameter_class=AlazarParameter, label='Clock Source', unit=None,
                           value='EXTERNAL_CLOCK_10_MHz_REF',
                           byte_to_value_dict={1: 'INTERNAL_CLOCK',
                                               4: 'SLOW_EXTERNAL_CLOCK',
                                               5: 'EXTERNAL_CLOCK_AC',
                                               7: 'EXTERNAL_CLOCK_10_MHz_REF'})
        # TODO (M) make parameter for board type

        # TODO (M) check board kind

# -----config-----
#         sample_rate
#         clock_edge
#         decimation
#
# coupling{n}
# range{n}
# impedence{n}
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


