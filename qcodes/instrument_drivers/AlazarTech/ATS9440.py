from .ATS import AlazarTech_ATS, AlazarParameter
from qcodes.utils import validators


class AlazarTech_ATS9440(AlazarTech_ATS):
    """
    This class is the driver for the ATS9440 board
    it inherits from the ATS base class

    TODO(nataliejpg):
        -  add clock source options and sample rate options
           (problem being that byte_to_value_dict of
           sample_rate relies on value of clock_source)

    """
    samples_divisor = 32
    channels = 4

    def __init__(self, name, **kwargs):
        dll_path = 'C:\\WINDOWS\\System32\\ATSApi.dll'
        super().__init__(name, dll_path=dll_path, **kwargs)

        # add parameters

        # ----- Parameters for the configuration of the board -----
        self.add_parameter(name='clock_source',
                           parameter_class=AlazarParameter,
                           label='Clock Source',
                           unit=None,
                           value='INTERNAL_CLOCK',
                           byte_to_value_dict={1: 'INTERNAL_CLOCK',
                                               2: 'FAST_EXTERNAL_CLOCK',
                                               4: 'SLOW_EXTERNAL_CLOCK',
                                               7: 'EXTERNAL_CLOCK_10MHz_REF'})
        self.add_parameter(name='external_sample_rate',
                           parameter_class=AlazarParameter,
                           label='External 10 MHz ref Sample Rate',
                           unit='S/s',
                           vals=validators.Enum(100000000, 125000000),
                           value=100000000)
        self.add_parameter(name='sample_rate',
                           parameter_class=AlazarParameter,
                           label='Internal Sample Rate',
                           unit='S/s',
                           value=100000000,
                           byte_to_value_dict={0x00000001: 1000,
                                                0x00000002: 2000,
                                                0x00000004: 5000,
                                                0x00000008: 10000,
                                                0x0000000A: 20000,
                                                0x0000000C: 50000,
                                                0x0000000E: 100000,
                                                0x00000010: 200000,
                                                0x00000012: 500000,
                                                0x00000014: 1000000,
                                                0x00000018: 2000000,
                                                0x0000001A: 5000000,
                                                0x0000001C: 10000000,
                                                0x0000001E: 20000000,
                                                0x00000022: 50000000,
                                                0x00000024: 100000000,
                                                0x00000025: 125000000,
                                                0x00000040: 'EXTERNAL_CLOCK',
                                                })
        self.add_parameter(name='manual_sample_rate',
                           get_cmd=None,
                           set_cmd=None,
                           label='External Sample Rate (manually set)',
                           unit='S/s',
                           vals=validators.Ints(0, 125000000),
                           initial_value=0)
        self.add_parameter(name='clock_edge',
                           parameter_class=AlazarParameter,
                           label='Clock Edge',
                           unit=None,
                           value='CLOCK_EDGE_RISING',
                           byte_to_value_dict={0: 'CLOCK_EDGE_RISING',
                                               1: 'CLOCK_EDGE_FALLING'})
        self.add_parameter(name='decimation',
                           parameter_class=AlazarParameter,
                           label='Decimation',
                           unit=None,
                           value=1,
                           vals=validators.Ints(0, 100000))

        for i in ['1', '2','3','4']:
            self.add_parameter(name='coupling' + i,
                               parameter_class=AlazarParameter,
                               label='Coupling channel ' + i,
                               unit=None,
                               value='DC',
                               byte_to_value_dict={1: 'AC', 2: 'DC'})
            self.add_parameter(name='channel_range' + i,
                               parameter_class=AlazarParameter,
                               label='Range channel ' + i,
                               unit='V',
                               value=4.0,
                               byte_to_value_dict={ 5: 0.1,
                                                    6: 0.2,
                                                    7: 0.4,
                                                    10: 1.0,
                                                    11: 2.0,
                                                    12: 4.0})
            self.add_parameter(name='impedance' + i,
                               parameter_class=AlazarParameter,
                               label='Impedance channel ' + i,
                               unit='Ohm',
                               value=50,
                               byte_to_value_dict={2: 50})

        self.add_parameter(name='trigger_operation',
                           parameter_class=AlazarParameter,
                           label='Trigger Operation',
                           unit=None,
                           value='TRIG_ENGINE_OP_J',
                           byte_to_value_dict={
                               0: 'TRIG_ENGINE_OP_J',
                               1: 'TRIG_ENGINE_OP_K',
                               2: 'TRIG_ENGINE_OP_J_OR_K',
                               3: 'TRIG_ENGINE_OP_J_AND_K',
                               4: 'TRIG_ENGINE_OP_J_XOR_K',
                               5: 'TRIG_ENGINE_OP_J_AND_NOT_K',
                               6: 'TRIG_ENGINE_OP_NOT_J_AND_K'})
        for i in ['1', '2']:
            self.add_parameter(name='trigger_engine' + i,
                               parameter_class=AlazarParameter,
                               label='Trigger Engine ' + i,
                               unit=None,
                               value='TRIG_ENGINE_' + ('J' if i == '1' else 'K'),
                               byte_to_value_dict={0: 'TRIG_ENGINE_J',
                                                   1: 'TRIG_ENGINE_K'})
            self.add_parameter(name='trigger_source' + i,
                               parameter_class=AlazarParameter,
                               label='Trigger Source ' + i,
                               unit=None,
                               value='EXTERNAL',
                               byte_to_value_dict={0: 'CHANNEL_A',
                                                   1: 'CHANNEL_B',
                                                   2: 'EXTERNAL',
                                                   3: 'DISABLE',
                                                   4: 'CHANNEL_C',
                                                   5: 'CHANNEL_D'})
            self.add_parameter(name='trigger_slope' + i,
                               parameter_class=AlazarParameter,
                               label='Trigger Slope ' + i,
                               unit=None,
                               value='TRIG_SLOPE_POSITIVE',
                               byte_to_value_dict={1: 'TRIG_SLOPE_POSITIVE',
                                                   2: 'TRIG_SLOPE_NEGATIVE'})
            self.add_parameter(name='trigger_level' + i,
                               parameter_class=AlazarParameter,
                               label='Trigger Level ' + i,
                               unit=None,
                               value=140,
                               vals=validators.Ints(0, 255))

        self.add_parameter(name='external_trigger_coupling',
                           parameter_class=AlazarParameter,
                           label='External Trigger Coupling',
                           unit=None,
                           value='DC',
                           byte_to_value_dict={1: 'AC', 2: 'DC'})
        self.add_parameter(name='external_trigger_range',
                           parameter_class=AlazarParameter,
                           label='External Trigger Range',
                           unit=None,
                           value='ETR_5V',
                           byte_to_value_dict={0: 'ETR_5V',
                                               2: 'ETR_TTL'})
        self.add_parameter(name='trigger_delay',
                           parameter_class=AlazarParameter,
                           label='Trigger Delay',
                           unit='Sample clock cycles',
                           value=0,
                           vals=validators.Ints(min_value=0))

        # NOTE: The board will wait for a for this amount of time for a
        # trigger event.  If a trigger event does not arrive, then the
        # board will automatically trigger. Set the trigger timeout value
        # to 0 to force the board to wait forever for a trigger event.
        #
        # IMPORTANT: The trigger timeout value should be set to zero after
        # appropriate trigger parameters have been determined, otherwise
        # the board may trigger if the timeout interval expires before a
        # hardware trigger event arrives.
        self.add_parameter(name='timeout_ticks',
                           parameter_class=AlazarParameter,
                           label='Timeout Ticks',
                           unit='10 us',
                           value=0,
                           vals=validators.Ints(min_value=0))

        self.add_parameter(name='aux_io_mode',
                           parameter_class=AlazarParameter,
                           label='AUX I/O Mode',
                           unit=None,
                           value='AUX_IN_AUXILIARY',
                           byte_to_value_dict={0: 'AUX_OUT_TRIGGER',
                                               1: 'AUX_IN_TRIGGER_ENABLE',
                                               13: 'AUX_IN_AUXILIARY'})

        self.add_parameter(name='aux_io_param',
                           parameter_class=AlazarParameter,
                           label='AUX I/O Param',
                           unit=None,
                           value='NONE',
                           byte_to_value_dict={0: 'NONE',
                                               1: 'TRIG_SLOPE_POSITIVE',
                                               2: 'TRIG_SLOPE_NEGATIVE'})

        # ----- Parameters for the acquire function -----
        self.add_parameter(name='mode',
                           parameter_class=AlazarParameter,
                           label='Acquisition mode',
                           unit=None,
                           value='NPT',
                           byte_to_value_dict={0x200: 'NPT', 0x400: 'TS'})
        self.add_parameter(name='samples_per_record',
                           parameter_class=AlazarParameter,
                           label='Samples per Record',
                           unit=None,
                           value=1024,
                           vals=validators.Multiples(
                                divisor=self.samples_divisor, min_value=256))
        self.add_parameter(name='records_per_buffer',
                           parameter_class=AlazarParameter,
                           label='Records per Buffer',
                           unit=None,
                           value=10,
                           vals=validators.Ints(min_value=0))
        self.add_parameter(name='buffers_per_acquisition',
                           parameter_class=AlazarParameter,
                           label='Buffers per Acquisition',
                           unit=None,
                           value=10,
                           vals=validators.Ints(min_value=0))
        self.add_parameter(name='channel_selection',
                           parameter_class=AlazarParameter,
                           label='Channel Selection',
                           unit=None,
                           value='AB',
                           byte_to_value_dict={1: 'A', 2: 'B', 3: 'AB',4: 'C',5: 'AC',6: 'BC',7: 'D',8: 'AD', 9: 'BD', 10: 'CD', 11: 'ABCD'})
        self.add_parameter(name='transfer_offset',
                           parameter_class=AlazarParameter,
                           label='Transfer Offset',
                           unit='Samples',
                           value=0,
                           vals=validators.Ints(min_value=0))
        self.add_parameter(name='external_startcapture',
                           parameter_class=AlazarParameter,
                           label='External Startcapture',
                           unit=None,
                           value='ENABLED',
                           byte_to_value_dict={0x0: 'DISABLED',
                                               0x1: 'ENABLED'})
        self.add_parameter(name='enable_record_headers',
                           parameter_class=AlazarParameter,
                           label='Enable Record Headers',
                           unit=None,
                           value='DISABLED',
                           byte_to_value_dict={0x0: 'DISABLED',
                                               0x8: 'ENABLED'})
        self.add_parameter(name='alloc_buffers',
                           parameter_class=AlazarParameter,
                           label='Alloc Buffers',
                           unit=None,
                           value='DISABLED',
                           byte_to_value_dict={0x0: 'DISABLED',
                                               0x20: 'ENABLED'})
        self.add_parameter(name='fifo_only_streaming',
                           parameter_class=AlazarParameter,
                           label='Fifo Only Streaming',
                           unit=None,
                           value='DISABLED',
                           byte_to_value_dict={0x0: 'DISABLED',
                                               0x800: 'ENABLED'})
        self.add_parameter(name='interleave_samples',
                           parameter_class=AlazarParameter,
                           label='Interleave Samples',
                           unit=None,
                           value='DISABLED',
                           byte_to_value_dict={0x0: 'DISABLED',
                                               0x1000: 'ENABLED'})
        self.add_parameter(name='get_processed_data',
                           parameter_class=AlazarParameter,
                           label='Get Processed Data',
                           unit=None,
                           value='DISABLED',
                           byte_to_value_dict={0x0: 'DISABLED',
                                               0x2000: 'ENABLED'})
        self.add_parameter(name='allocated_buffers',
                           parameter_class=AlazarParameter,
                           label='Allocated Buffers',
                           unit=None,
                           value=4,
                           vals=validators.Ints(min_value=0))

        self.add_parameter(name='buffer_timeout',
                           parameter_class=AlazarParameter,
                           label='Buffer Timeout',
                           unit='ms',
                           value=1000,
                           vals=validators.Ints(min_value=0))

        model = self.get_idn()['model']
        if model != 'ATS9440':
            raise Exception("The Alazar board kind is not 'ATS9440',"
                            " found '" + str(model) + "' instead.")
