from .ATS import AlazarTech_ATS, AlazarParameter
from qcodes.utils import validators


class AlazarTech_ATS9360(AlazarTech_ATS):
    # TODO(nataliejpg) add more options for the sample rate set using 10MHz ref
    # including so that the 'get sample rate' function in ATS.py works
    def __init__(self, name, **kwargs):
        dll_path = 'C:\\WINDOWS\\System32\\ATSApi.dll'
        super().__init__(name, dll_path=dll_path, **kwargs)

        # add parameters

        # ----- Parameters for the configuration of the board -----
        self.add_parameter(name='clock_source',
                           parameter_class=AlazarParameter,
                           label='Clock Source',
                           unit=None,
                           value='EXTERNAL_CLOCK_10MHz_REF',
                           byte_to_value_dict={1: 'INTERNAL_CLOCK',
                                               4: 'SLOW_EXTERNAL_CLOCK',
                                               5: 'EXTERNAL_CLOCK_AC',
                                               7: 'EXTERNAL_CLOCK_10MHz_REF'})
        self.add_parameter(name='sample_rate',
                           parameter_class=AlazarParameter,
                           label='Sample Rate',
                           unit='S/s',
                           value='10MHZ_REF_500MSPS',
                           byte_to_value_dict={
                               0x1: '1KSPS', 0x2: '2KSPS',
                               0x4: '5KSPS', 0x8: '10KSPS',
                               0xA: '20KSPS', 0xC: '50KSPS',
                               0xE: '100KSPS', 0x10: '200KSPS',
                               0x12: '500KSPS', 0x14: '1MSPS',
                               0x18: '2MSPS', 0x1A: '5MSPS',
                               0x1C: '10MSPS', 0x1E: '20MSPS',
                               0x22: '50MSPS', 0x24: '100MSPS',
                               0x25: '125MSPS', 0x2B: '250MSPS',
                               0x30: '500MSPS', 0x32: '800MSPS',
                               0x35: '1000MSPS', 0x37: '1200MSPS',
                               0x3A: '1500MSPS', 0x3D: '1800MSPS',
                               0x40: 'EXTERNAL_CLOCK',
                               500000000: '10MHZ_REF_500MSPS'})
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

        for i in ['1', '2']:
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
                               value=0.4,
                               byte_to_value_dict={7: 0.4})
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
                               value='TRIG_ENGINE_' + ('J' if i == 0 else 'K'),
                               byte_to_value_dict={0: 'TRIG_ENGINE_J',
                                                   1: 'TRIG_ENGINE_K'})
            self.add_parameter(name='trigger_source' + i,
                               parameter_class=AlazarParameter,
                               label='Trigger Source ' + i,
                               unit=None,
                               value='DISABLE',
                               byte_to_value_dict={0: 'CHANNEL_A',
                                                   1: 'CHANNEL_B',
                                                   2: 'EXTERNAL',
                                                   3: 'DISABLE'})
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
                           value='ETR_2V5',
                           byte_to_value_dict={0: 'ETR_5V', 1: 'ETR_1V',
                                               2: 'ETR_TTL', 3: 'ETR_2V5'})
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
                           unit=None,
                           value='NONE',
                           byte_to_value_dict={0: 'NONE',
                                               1: 'TRIG_SLOPE_POSITIVE',
                                               2: 'TRIG_SLOPE_NEGATIVE'})

        # ----- Parameters for the acquire function -----
        self.add_parameter(name='mode',
                           parameter_class=AlazarParameter,
                           label='Acquisiton mode',
                           unit=None,
                           value='NPT',
                           byte_to_value_dict={0x200: 'NPT', 0x400: 'TS'})
        self.add_parameter(name='samples_per_record',
                           parameter_class=AlazarParameter,
                           label='Samples per Record',
                           unit=None,
                           value=1024,
                           vals=validators.Multiples(
                                divisor=64, min_value=256))
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
                           byte_to_value_dict={1: 'A', 2: 'B', 3: 'AB'})
        self.add_parameter(name='transfer_offset',
                           parameter_class=AlazarParameter,
                           label='Transer Offset',
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
                           value=2,
                           vals=validators.Ints(min_value=0))
        self.add_parameter(name='buffer_timeout',
                           parameter_class=AlazarParameter,
                           label='Buffer Timeout',
                           unit='ms',
                           value=1000,
                           vals=validators.Ints(min_value=0))

        

        model = self.get_idn()['model']
        if model != 'ATS9360':
            raise Exception("The Alazar board kind is not 'ATS9360',"
                            " found '" + str(model) + "' instead.")
