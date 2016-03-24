from .ATS import AlazarTech_ATS


class AlazarTech_ATS9870(AlazarTech_ATS):
    def __init__(self, name):
        super().__init__(name)
        # TODO (M) check board kind
        # add parameters
        # TODO (M) make parameter for board type

# -----config-----
# clock source
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


