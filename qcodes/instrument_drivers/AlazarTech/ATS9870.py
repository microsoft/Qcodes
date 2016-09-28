from .ATS import AlazarTech_ATS, AlazarParameter
from qcodes.utils import validators


class AlazarTech_ATS9870(AlazarTech_ATS):
    """
    This class is the driver for the ATS9870 board
    it inherits from the ATS base class

    it creates all necessary parameters for the Alazar card
    """
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
                                               4: 'SLOW_EXTERNAL_CLOCK',
                                               5: 'EXTERNAL_CLOCK_AC',
                                               7: 'EXTERNAL_CLOCK_10_MHz_REF'})
        self.add_parameter(name='sample_rate',
                           parameter_class=AlazarParameter,
                           label='Sample Rate',
                           unit='S/s',
                           value=1000000000,
                           byte_to_value_dict={
                               0x1: 1000, 0x2: 2000, 0x4: 5000, 0x8: 10000,
                               0xA: 20000, 0xC: 50000, 0xE: 100000,
                               0x10: 200000, 0x12: 500000, 0x14: 1000000,
                               0x18: 2000000, 0x1A: 5000000, 0x1C: 10000000,
                               0x1E: 20000000, 0x22: 50000000, 0x24: 100000000,
                               0x2B: 250000000, 0x30: 500000000,
                               0x35: 1000000000, 0x40: 'EXTERNAL_CLOCK',
                               1000000000: '1GHz_REFERENCE_CLOCK'})
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
                           value=0,
                           vals=validators.Ints(0, 100000))

        for i in ['1', '2']:
            self.add_parameter(name='coupling' + i,
                               parameter_class=AlazarParameter,
                               label='Coupling channel ' + i,
                               unit=None,
                               value='AC',
                               byte_to_value_dict={1: 'AC', 2: 'DC'})
            self.add_parameter(name='channel_range' + i,
                               parameter_class=AlazarParameter,
                               label='Range channel ' + i,
                               unit='V',
                               value=4,
                               byte_to_value_dict={
                                   2: 0.04, 5: 0.1, 6: 0.2, 7: 0.4,
                                   10: 1., 11: 2., 12: 4.})
            self.add_parameter(name='impedance' + i,
                               parameter_class=AlazarParameter,
                               label='Impedance channel ' + i,
                               unit='Ohm',
                               value=50,
                               byte_to_value_dict={1: 1000000, 2: 50})
            self.add_parameter(name='bwlimit' + i,
                               parameter_class=AlazarParameter,
                               label='Bandwidth limit channel ' + i,
                               unit=None,
                               value='DISABLED',
                               byte_to_value_dict={0: 'DISABLED',
                                                   1: 'ENABLED'})

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
                               value=128,
                               vals=validators.Ints(0, 255))

        self.add_parameter(name='external_trigger_coupling',
                           parameter_class=AlazarParameter,
                           label='External Trigger Coupling',
                           unit=None,
                           value='AC',
                           byte_to_value_dict={1: 'AC', 2: 'DC'})
        self.add_parameter(name='external_trigger_range',
                           parameter_class=AlazarParameter,
                           label='External Trigger Range',
                           unit=None,
                           value='ETR_5V',
                           byte_to_value_dict={0: 'ETR_5V', 1: 'ETR_1V'})
        self.add_parameter(name='trigger_delay',
                           parameter_class=AlazarParameter,
                           label='Trigger Delay',
                           unit='Sample clock cycles',
                           value=0,
                           vals=validators.Ints(min_value=0))
        self.add_parameter(name='timeout_ticks',
                           parameter_class=AlazarParameter,
                           label='Timeout Ticks',
                           unit='10 us',
                           value=0,
                           vals=validators.Ints(min_value=0))

        # ----- Parameters for the acquire function -----
        self.add_parameter(name='mode',
                           parameter_class=AlazarParameter,
                           label='Acquisiton mode',
                           unit=None,
                           value='NPT',
                           byte_to_value_dict={0x200: 'NPT', 0x400: 'TS'})

        # samples_per_record must be a multiple of of some number (64 in the
        # case of ATS9870) and and has some minimum (256 in the case of ATS9870)
        # These values can be found in the ATS-SDK programmar's guide
        self.add_parameter(name='samples_per_record',
                           parameter_class=AlazarParameter,
                           label='Samples per Record',
                           unit=None,
                           value=96000,
                           vals=validators.Multiples(divisor=64, min_value=256))

        # TODO(damazter) (M) figure out if this also has to be a multiple of
        # something,
        # I could not find this in the documentation but somehow I have the
        # feeling it still should be a multiple of something
        # NOTE by ramiro: At least in previous python implementations
        # (PycQED delft), this is an artifact for compatibility with
        # AWG sequencing, not particular to any ATS architecture.
        #                  ==> this is a construction imposed by the memory
        #                      strategy implemented on the python driver we
        #                      are writing, not limited by any actual ATS
        #                      feature.
        self.add_parameter(name='records_per_buffer',
                           parameter_class=AlazarParameter,
                           label='Records per Buffer',
                           unit=None,
                           value=1,
                           vals=validators.Ints(min_value=0))
        self.add_parameter(name='buffers_per_acquisition',
                           parameter_class=AlazarParameter,
                           label='Buffers per Acquisition',
                           unit=None,
                           value=1,
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
                           value=1,
                           vals=validators.Ints(min_value=0))
        self.add_parameter(name='buffer_timeout',
                           parameter_class=AlazarParameter,
                           label='Buffer Timeout',
                           unit='ms',
                           value=1000,
                           vals=validators.Ints(min_value=0))

        model = self.get_idn()['model']
        if model != 'ATS9870':
            raise Exception("The Alazar board kind is not 'ATS9870',"
                            " found '" + str(model) + "' instead.")