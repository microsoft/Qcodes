from .ATS import AlazarTech_ATS, AlazarParameter
from qcodes.utils import validators


class ATS9440(AlazarTech_ATS):
    def __init__(self, name, **kwargs):
        dll_path = 'C:\\WINDOWS\\System32\\ATSApi.dll'
        super().__init__(name, dll_path=dll_path, **kwargs)

        self._bwlimit_support = False

        # add parameters
        self.channels = ['A', 'B', 'C', 'D']

        # ----- Parameters for the configuration of the board -----
        self.add_parameter(name='clock_source',
                           parameter_class=AlazarParameter,
                           label='Clock Source',
                           unit=None,
                           value='internal_clock',
                           byte_to_value_dict={1: 'internal_clock',
                                               4: 'slow_external_clock',
                                               5: 'external_clock_AC',
                                               7: 'external_clock_10_MHz_ref'})
        self.add_parameter(name='sample_rate',
                           parameter_class=AlazarParameter,
                           label='Sample Rate',
                           unit='S/s',
                           value=100000,
                           byte_to_value_dict={
                               0x1: 1000, 0x2: 2000, 0x4: 5000, 0x8: 10000,
                               0xA: 20000, 0xC: 50000, 0xE: 100000,
                               0x10: 200000, 0x12: 500000, 0x14: 1000000,
                               0x18: 2000000, 0x1A: 5000000, 0x1C: 10000000,
                               0x1E: 20000000, 0x22: 50000000, 0x24: 100000000,
                               0x25: 125000000, 0x40: 'external_clock',
                               1000000000: '1GHz_reference_clock'})
        self.add_parameter(name='clock_edge',
                           parameter_class=AlazarParameter,
                           label='Clock Edge',
                           unit=None,
                           value='rising',
                           byte_to_value_dict={0: 'rising',
                                               1: 'falling'})

        self.add_parameter(name='decimation',
                           parameter_class=AlazarParameter,
                           label='Decimation',
                           unit=None,
                           value=0,
                           vals=validators.Ints(0, 100000))

        # Acquisition channel parameters
        for ch in self.channels:
            self.add_parameter(name='coupling' + ch,
                               parameter_class=AlazarParameter,
                               label='Coupling channel ' + ch,
                               unit=None,
                               value='DC',
                               byte_to_value_dict={1: 'AC', 2: 'DC'})
            self.add_parameter(name='channel_range' + ch,
                               parameter_class=AlazarParameter,
                               label='Range channel ' + ch,
                               unit='V',
                               value=1,
                               byte_to_value_dict={
                                   5: 0.1, 6: 0.2, 7: 0.4,
                                   10: 1., 11: 2., 12: 4.})
            self.add_parameter(name='impedance' + ch,
                               parameter_class=AlazarParameter,
                               label='Impedance channel ' + ch,
                               unit='Ohm',
                               value=50,
                               byte_to_value_dict={2: 50})

        # Trigger parameters
        self.add_parameter(name='trigger_operation',
                           parameter_class=AlazarParameter,
                           label='Trigger Operation',
                           unit=None,
                           value='J',
                           byte_to_value_dict={
                               0: 'J',
                               1: 'K',
                               2: 'J_or_K',
                               3: 'J_and_K',
                               4: 'J_xor_K',
                               5: 'J_and_not_K',
                               6: 'not_J_and_K'})
        for i in ['1', '2']:
            self.add_parameter(name='trigger_engine' + i,
                               parameter_class=AlazarParameter,
                               label='Trigger Engine ' + i,
                               unit=None,
                               value=('J' if i == '1' else 'K'),
                               byte_to_value_dict={0: 'J',
                                                   1: 'K'})
            self.add_parameter(name='trigger_source' + i,
                               parameter_class=AlazarParameter,
                               label='Trigger Source ' + i,
                               unit=None,
                               value='disable',
                               byte_to_value_dict={0: 'A',
                                                   1: 'B',
                                                   2: 'trig_in',
                                                   3: 'disable',
                                                   4: 'C',
                                                   5: 'D'})
            self.add_parameter(name='trigger_slope' + i,
                               parameter_class=AlazarParameter,
                               label='Trigger Slope ' + i,
                               unit=None,
                               value='positive',
                               byte_to_value_dict={1: 'positive',
                                                   2: 'negative'})
            self.add_parameter(name='trigger_level' + i,
                               parameter_class=AlazarParameter,
                               label='Trigger Level ' + i,
                               unit=None,
                               value=150,
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
                           unit='V',
                           value=5,
                           byte_to_value_dict={0: 5, 1: 1})
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

        # ----- Parameters for the acquire function -----
        self.add_parameter(name='mode',
                           parameter_class=AlazarParameter,
                           label='Acquisition mode',
                           unit=None,
                           value='NPT',
                           byte_to_value_dict={0x100: 'CS', 0x200: 'NPT',
                                               0x400: 'TS'})

        # samples_per_record must be a multiple of 32, and 256 minimum!
        # TODO check if it is 32 or 16, manual is unclear
        self.add_parameter(name='samples_per_record',
                           parameter_class=AlazarParameter,
                           label='Samples per Record',
                           unit=None,
                           value=1024,
                           vals=validators.Multiples(divisor=16, min_value=16))

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
                           byte_to_value_dict={1: 'A', 2: 'B', 3: 'AB', 4:'C',
                                               5: 'AC', 6: 'BC',8:'D', 9: 'AD',
                                               10: 'BD', 12: 'CD', 15: 'ABCD'})
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
                           value='enabled',
                           byte_to_value_dict={0x0: 'disabled',
                                               0x1: 'enabled'})
        self.add_parameter(name='enable_record_headers',
                           parameter_class=AlazarParameter,
                           label='Enable Record Headers',
                           unit=None,
                           value='disabled',
                           byte_to_value_dict={0x0: 'disabled',
                                               0x8: 'enabled'})
        self.add_parameter(name='alloc_buffers',
                           parameter_class=AlazarParameter,
                           label='Alloc Buffers',
                           unit=None,
                           value='disabled',
                           byte_to_value_dict={0x0: 'disabled',
                                               0x20: 'enabled'})
        self.add_parameter(name='fifo_only_streaming',
                           parameter_class=AlazarParameter,
                           label='Fifo Only Streaming',
                           unit=None,
                           value='disabled',
                           byte_to_value_dict={0x0: 'disabled',
                                               0x800: 'enabled'})
        self.add_parameter(name='interleave_samples',
                           parameter_class=AlazarParameter,
                           label='Interleave Samples',
                           unit=None,
                           value='disabled',
                           byte_to_value_dict={0x0: 'disabled',
                                               0x1000: 'enabled'})
        self.add_parameter(name='get_processed_data',
                           parameter_class=AlazarParameter,
                           label='Get Processed Data',
                           unit=None,
                           value='disabled',
                           byte_to_value_dict={0x0: 'disabled',
                                               0x2000: 'enabled'})

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

        # TODO (M) make parameter for board type

        # TODO (M) check board kind
