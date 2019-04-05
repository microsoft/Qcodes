from .ATS import AlazarTech_ATS
from .utils import TraceParameter
from qcodes.utils import validators


class AlazarTech_ATS9870(AlazarTech_ATS):
    """
    This class is the driver for the ATS9870 board
    it inherits from the ATS base class

    It creates all necessary parameters for the Alazar card
    """
    def __init__(self, name, **kwargs):
        dll_path = 'C:\\WINDOWS\\System32\\ATSApi.dll'
        super().__init__(name, dll_path=dll_path, **kwargs)
        # add parameters

        # ----- Parameters for the configuration of the board -----
        self.add_parameter(name='clock_source',
                           parameter_class=TraceParameter,
                           label='Clock Source',
                           unit=None,
                           initial_value='INTERNAL_CLOCK',
                           val_mapping={'INTERNAL_CLOCK': 1,
                                        'SLOW_EXTERNAL_CLOCK': 4,
                                        'EXTERNAL_CLOCK_AC': 5,
                                        'EXTERNAL_CLOCK_10_MHz_REF': 7})
        
        self.add_parameter(name='external_sample_rate',
                           get_cmd=None,
                           parameter_class=TraceParameter,
                           label='External Sample Rate',
                           unit='S/s',
                           vals=validators.Enum(1_000_000_000),
                           initial_value=1_000_000_000)
        
        self.add_parameter(name='sample_rate',
                           parameter_class=TraceParameter,
                           label='Sample Rate',
                           unit='S/s',
                           initial_value=1000000000,
                           val_mapping={
                               1000: 0x1,
                               2000: 0x2,
                               5000: 0x4,
                               10000: 0x8,
                               20000: 0xA,
                               50000: 0xC,
                               100000: 0xE,
                               200000: 0x10,
                               500000: 0x12,
                               1000000: 0x14,
                               2000000: 0x18,
                               5000000: 0x1A,
                               10000000: 0x1C,
                               20000000: 0x1E,
                               50000000: 0x22,
                               100000000: 0x24,
                               250000000: 0x2B,
                               500000000: 0x30,
                               1000000000: 0x35,
                               'EXTERNAL_CLOCK': 0x40,
                               '1GHz_REFERENCE_CLOCK': 1000000000})

        self.add_parameter(name='clock_edge',
                           parameter_class=TraceParameter,
                           label='Clock Edge',
                           unit=None,
                           initial_value='CLOCK_EDGE_RISING',
                           val_mapping={'CLOCK_EDGE_RISING': 0,
                                        'CLOCK_EDGE_FALLING': 1})

        self.add_parameter(name='decimation',
                           parameter_class=TraceParameter,
                           label='Decimation',
                           unit=None,
                           initial_value=0,
                           vals=validators.Ints(0, 100000))

        for i in ['1', '2']:
            self.add_parameter(name='coupling' + i,
                               parameter_class=TraceParameter,
                               label='Coupling channel ' + i,
                               unit=None,
                               initial_value='AC',
                               val_mapping={'AC': 1, 'DC': 2})
            self.add_parameter(name='channel_range' + i,
                               parameter_class=TraceParameter,
                               label='Range channel ' + i,
                               unit='V',
                               initial_value=4,
                               val_mapping={0.04: 2,
                                            0.1: 5,
                                            0.2: 6,
                                            0.4: 7,
                                            1.0: 10,
                                            2.0: 11,
                                            4.0: 12})
            self.add_parameter(name='impedance' + i,
                               parameter_class=TraceParameter,
                               label='Impedance channel ' + i,
                               unit='Ohm',
                               initial_value=50,
                               val_mapping={1000000: 1,
                                            50: 2})
            self.add_parameter(name='bwlimit' + i,
                               parameter_class=TraceParameter,
                               label='Bandwidth limit channel ' + i,
                               unit=None,
                               initial_value='DISABLED',
                               val_mapping={'DISABLED': 0,
                                            'ENABLED': 1})

        self.add_parameter(name='trigger_operation',
                           parameter_class=TraceParameter,
                           label='Trigger Operation',
                           unit=None,
                           initial_value='TRIG_ENGINE_OP_J',
                           val_mapping={'TRIG_ENGINE_OP_J': 0,
                                        'TRIG_ENGINE_OP_K': 1,
                                        'TRIG_ENGINE_OP_J_OR_K': 2,
                                        'TRIG_ENGINE_OP_J_AND_K': 3,
                                        'TRIG_ENGINE_OP_J_XOR_K': 4,
                                        'TRIG_ENGINE_OP_J_AND_NOT_K': 5,
                                        'TRIG_ENGINE_OP_NOT_J_AND_K': 6})
        for i in ['1', '2']:
            self.add_parameter(name='trigger_engine' + i,
                               parameter_class=TraceParameter,
                               label='Trigger Engine ' + i,
                               unit=None,
                               initial_value='TRIG_ENGINE_' + ('J' if i == 0 else 'K'),
                               val_mapping={'TRIG_ENGINE_J': 0,
                                            'TRIG_ENGINE_K': 1})
            self.add_parameter(name='trigger_source' + i,
                               parameter_class=TraceParameter,
                               label='Trigger Source ' + i,
                               unit=None,
                               initial_value='DISABLE',
                               val_mapping={'CHANNEL_A': 0,
                                            'CHANNEL_B': 1,
                                            'EXTERNAL': 2,
                                            'DISABLE': 3})
            self.add_parameter(name='trigger_slope' + i,
                               parameter_class=TraceParameter,
                               label='Trigger Slope ' + i,
                               unit=None,
                               initial_value='TRIG_SLOPE_POSITIVE',
                               val_mapping={'TRIG_SLOPE_POSITIVE': 1,
                                            'TRIG_SLOPE_NEGATIVE': 2})
            self.add_parameter(name='trigger_level' + i,
                               parameter_class=TraceParameter,
                               label='Trigger Level ' + i,
                               unit=None,
                               initial_value=128,
                               vals=validators.Ints(0, 255))

        self.add_parameter(name='external_trigger_coupling',
                           parameter_class=TraceParameter,
                           label='External Trigger Coupling',
                           unit=None,
                           initial_value='AC',
                           val_mapping={'AC': 1,
                                        'DC': 2})
        self.add_parameter(name='external_trigger_range',
                           parameter_class=TraceParameter,
                           label='External Trigger Range',
                           unit=None,
                           initial_value='ETR_5V',
                           val_mapping={'ETR_5V': 0,
                                        'ETR_1V': 1})
        self.add_parameter(name='trigger_delay',
                           parameter_class=TraceParameter,
                           label='Trigger Delay',
                           unit='Sample clock cycles',
                           initial_value=0,
                           vals=validators.Ints(min_value=0))
        self.add_parameter(name='timeout_ticks',
                           parameter_class=TraceParameter,
                           label='Timeout Ticks',
                           unit='10 us',
                           initial_value=0,
                           vals=validators.Ints(min_value=0))
        self.add_parameter(name='aux_io_mode',
                           parameter_class=TraceParameter,
                           label='AUX I/O Mode',
                           unit=None,
                           initial_value='AUX_IN_AUXILIARY',
                           val_mapping={'AUX_OUT_TRIGGER': 0,
                                        'AUX_IN_TRIGGER_ENABLE': 1,
                                        'AUX_IN_AUXILIARY': 13})

        self.add_parameter(name='aux_io_param',
                           parameter_class=TraceParameter,
                           label='AUX I/O Param',
                           unit=None,
                           initial_value='NONE',
                           val_mapping={'NONE': 0,
                                        'TRIG_SLOPE_POSITIVE': 1,
                                        'TRIG_SLOPE_NEGATIVE': 2})


        # ----- Parameters for the acquire function -----
        self.add_parameter(name='mode',
                           label='Acquisition mode',
                           unit=None,
                           initial_value='NPT',
                           get_cmd=None,
                           set_cmd=None,
                           val_mapping={'NPT': 0x200, 'TS': 0x400})

        # samples_per_record must be a multiple of of some number (64 in the
        # case of ATS9870) and and has some minimum (256 in the case of ATS9870)
        # These values can be found in the ATS-SDK programmar's guide
        self.add_parameter(name='samples_per_record',
                           label='Samples per Record',
                           unit=None,
                           initial_value=96000,
                           get_cmd=None,
                           set_cmd=None,
                           vals=validators.Multiples(
                                divisor=64, min_value=256))

        self.add_parameter(name='records_per_buffer',
                           label='Records per Buffer',
                           unit=None,
                           initial_value=1,
                           get_cmd=None,
                           set_cmd=None,
                           vals=validators.Ints(min_value=0))
        self.add_parameter(name='buffers_per_acquisition',
                           label='Buffers per Acquisition',
                           unit=None,
                           get_cmd=None,
                           set_cmd=None,
                           initial_value=1,
                           vals=validators.Ints(min_value=0))
        self.add_parameter(name='channel_selection',
                           label='Channel Selection',
                           unit=None,
                           get_cmd=None,
                           set_cmd=None,
                           initial_value='AB',
                           val_mapping={'A': 1, 'B': 2, 'AB': 3})
        self.add_parameter(name='transfer_offset',
                           label='Transfer Offset',
                           unit='Samples',
                           get_cmd=None,
                           set_cmd=None,
                           initial_value=0,
                           vals=validators.Ints(min_value=0))
        self.add_parameter(name='external_startcapture',
                           label='External Startcapture',
                           unit=None,
                           get_cmd=None,
                           set_cmd=None,
                           initial_value='ENABLED',
                           val_mapping={'DISABLED': 0X0,
                                        'ENABLED': 0x1})
        self.add_parameter(name='enable_record_headers',
                           label='Enable Record Headers',
                           unit=None,
                           get_cmd=None,
                           set_cmd=None,
                           initial_value='DISABLED',
                           val_mapping={'DISABLED': 0x0,
                                        'ENABLED': 0x8})
        self.add_parameter(name='alloc_buffers',
                           label='Alloc Buffers',
                           unit=None,
                           get_cmd=None,
                           set_cmd=None,
                           initial_value='DISABLED',
                           val_mapping={'DISABLED': 0x0,
                                        'ENABLED': 0x20})
        self.add_parameter(name='fifo_only_streaming',
                           label='Fifo Only Streaming',
                           unit=None,
                           get_cmd=None,
                           set_cmd=None,
                           initial_value='DISABLED',
                           val_mapping={'DISABLED': 0x0,
                                        'ENABLED': 0x800})
        self.add_parameter(name='interleave_samples',
                           label='Interleave Samples',
                           unit=None,
                           get_cmd=None,
                           set_cmd=None,
                           initial_value='DISABLED',
                           val_mapping={'DISABLED': 0x0,
                                        'ENABLED': 0x1000})
        self.add_parameter(name='get_processed_data',
                           label='Get Processed Data',
                           unit=None,
                           get_cmd=None,
                           set_cmd=None,
                           initial_value='DISABLED',
                           val_mapping={'DISABLED': 0x0,
                                        'ENABLED': 0x2000})
        self.add_parameter(name='allocated_buffers',
                           label='Allocated Buffers',
                           unit=None,
                           get_cmd=None,
                           set_cmd=None,
                           initial_value=1,
                           vals=validators.Ints(min_value=0))
        self.add_parameter(name='buffer_timeout',
                           label='Buffer Timeout',
                           unit='ms',
                           get_cmd=None,
                           set_cmd=None,
                           initial_value=1000,
                           vals=validators.Ints(min_value=0))

        model = self.get_idn()['model']
        if model != 'ATS9870':
            raise Exception("The Alazar board kind is not 'ATS9870',"
                            " found '" + str(model) + "' instead.")
