import array
import warnings

from qcodes import VisaInstrument, InstrumentChannel, ChannelList, \
    StandardParameter, ManualParameter
from qcodes import validators as vals


class EnumVisa(vals.Enum):
    def __init__(self, *values):
        values_lowercase = [val.lower() for val in values]
        super().__init__(*values_lowercase)

        self.val_mapping = {val.lower(): ''.join(c for c in val if c.isupper())
                            for val in values}


class ChannelVisaParameter(StandardParameter):
    def __init__(self, name, channel_id,
                 vals=None, val_mapping=None, **kwargs):
        if isinstance(vals, EnumVisa):
            val_mapping = vals.val_mapping

        super().__init__(name, vals=vals, val_mapping=val_mapping, **kwargs)

        self.channel_id = channel_id

    def get(self):
        # Set active channel to current channel if necessary
        active_channel = self._instrument._parent.active_channel.get_latest()
        if active_channel != self.channel_id:
            self._instrument._parent.active_channel(self.channel_id)
        return super().get()


class AWGChannel(InstrumentChannel):
    def __init__(self, parent, name, id, **kwargs):
        super().__init__(parent, name, **kwargs)

        self.id = id

        self.write = self._parent.write
        self.visa_handle = self._parent.visa_handle

        self.add_parameter(
            'enable_mode',
            get_cmd='INITIATE:CONTINUOUS:ENABLE?',
            set_cmd='INITIATE:CONTINUOUS:ENABLE {}',
            vals=EnumVisa('SELF', 'ARMed'),
            docstring='Possible enable modes are: '
                      'self: In continuous run mode, waveforms are generated '
                      'at the output connector as soon as they are selected. '
                      'armed: The 81180A generates waveforms at the output '
                      'connector only after calling enable.')

        self.add_parameter(
            'sample_rate',
            get_cmd='FREQUENCY:RASTER?',
            set_cmd='FREQUENCY:RASTER {:.8f}',
            get_parser=float,
            vals=vals.Numbers(10e6, 4.2e9),
            docstring='Set the sample rate of the arbitrary waveforms. Has no '
                      'effect on standard waveforms')

        self.add_parameter(
            'frequency_standard_waveforms',
            unit='Hz',
            get_cmd='FREQUENCY?',
            set_cmd='FREQUENCY {:.8f}',
            get_parser=float,
            vals=vals.Numbers(10e-3, 250e6),
            docstring='Set the frequency of standard waveforms')

        self.add_parameter(
            'run_mode',
            get_cmd='FUNCTION:MODE?',
            set_cmd='FUNCTION:MODE {}',
            vals=EnumVisa('FIXed', 'USER', 'SEQuenced', 'ASEQuenced',
                          'MODulated', 'PULSe'),
            docstring='Run mode defines the type of waveform that is '
                      'available. Possible run modes are: '
                      'fixed: standard waveform shapes. '
                      'user: arbitrary waveform shapes. '
                      'sequenced: sequence of arbitrary waveforms. '
                      'asequenced: advanced sequence of arbitrary waveforms. '
                      'modulated: modulated (standard) waveforms. '
                      'pulse: digital pulse function')

        self.add_parameter(
            'output_coupling',
            get_cmd='OUTPUT:COUPLING?',
            set_cmd='OUTPUT:COUPLING {}',
            vals=vals.Enum('DC, DAC', 'AC'),
            docstring='Possible output couplings are: '
                      'DC: optimized for pulse response at high amplitudes. '
                      'DAC: optimized for bandwidth but low amplitude. '
                      'AC: optimized for bandwidth.'
                      'Note that DC and DAC use the DC output, and AC uses '
                      'the AC output. DC and DAC couplings also allow '
                      'control of amplitude and offset')

        self.add_parameter(
            'output_state',
            get_cmd='OUTPUT?',
            set_cmd='OUTPUT {}',
            vals=EnumVisa('ON', 'OFF'))

        self.add_parameter(
            'power',
            unit='dBm',
            get_cmd='POWER?',
            set_cmd='POWER {:.8f}',
            get_parser=float,
            vals=vals.Numbers(-8, 8), # Might be -5 to 5, manual is unclear
            docstring='Set output power from AC output path (50-ohm matched)')

        self.add_parameter(
            'voltage_DAC',
            unit='V',
            get_cmd='VOLTAGE?',
            set_cmd='VOLTAGE {:.8f}',
            get_parser=float,
            vals=vals.Numbers(50e-3, 2),
            docstring='Waveform amplitude when routed through the DC path')

        self.add_parameter(
            'voltage_DC',
            unit='V',
            get_cmd='VOLTAGE:DAC?',
            set_cmd='VOLTAGE:DAC {:.8f}',
            get_parser=float,
            vals=vals.Numbers(50e-3, 2),
            docstring='Waveform amplitude when routed through the DAC path')

        self.add_parameter(
            'voltage_offset',
            unit='V',
            get_cmd='VOLTAGE:OFFSET?',
            set_cmd='VOLTAGE:OFFSET {:.8f}',
            get_parser=float,
            vals=vals.Numbers(-1.5, 1.5),
            docstring='Voltage offset when routed through DAC or DC path')


        # Trigger parameters
        self.add_parameter(
            'trigger_input',
            set_cmd='TRIGGER:{}',
            vals=vals.Enum('ECL', 'TTL'),
            docstring='Set trigger input. Possible inputs are'
                      'ECL: negative signal at fixed -1.3V.'
                      'TTL: variable trigger_level')

        self.add_parameter(
            'trigger_source',
            get_cmd='TRIGGER:SOURCE:ADVANCE?',
            set_cmd='TRIGGER:SOURCE:ADVANCE {}',
            vals=EnumVisa('EXTernal', 'BUS', 'TIMer', 'EVENt'))

        self.add_parameter(
            'trigger_mode',
            get_cmd='TRIGGER:MODE?',
            set_cmd='TRIGGER:MODE {}',
            vals=EnumVisa('NORMal', 'OVERride'),
            docstring='Possible trigger modes are:'
                      'normal:  the first trigger activates the output and '
                      'consecutive triggers are ignored for the duration of '
                      'the output waveform. '
                      'override:  the first trigger activates  the output '
                      'and consecutive triggers  restart the output waveform, '
                      'regardless if the current waveform has been completed '
                      'or not.')

        self.add_parameter(
            'trigger_timer_mode',
            get_cmd='TRIGGER:TIMER:MODE?',
            set_cmd='TRIGGER:TIMER:MODE {}',
            vals=EnumVisa('TIME', 'DELay'),
            docstring='Possible modes of internal trigger are:'
                      'time: Perform trigger at fixed time interval.'
                      'delay: Perform trigger at fixed time intervals after '
                      'previous waveform finished.')

        self.add_parameter(
            'trigger_timer_delay',
            get_cmd='TRIGGER:TIMER:DELAY?',
            set_cmd='TRIGGER:TIMER:DELAY {:d}',
            get_parser=int,
            unit='1/sample rate',
            vals=vals.Multiples(min_value=152, max_value=8000000, divisor=8))

        self.add_parameter(
            'trigger_level',
            get_cmd='TRIGGER:LEVEL?',
            set_cmd='TRIGGER:LEVEL {:.4f}',
            get_parser=float,
            vals=vals.Numbers(-5, 5),
            unit='V',
            docstring='Trigger level when trigger_mode is TTL')

        self.add_parameter(
            'trigger_slope',
            get_cmd='TRIGGER:SLOPE?',
            set_cmd='TRIGGER:SLOPE {}',
            vals=EnumVisa('POSitive', 'NEGative', 'EITher'))

        self.add_parameter(
            'trigger_delay',
            get_cmd='TRIGGER:DELAY?',
            set_cmd='TRIGGER:DELAY {.4f}',
            get_parser=float,
            unit='1/sample rate',
            vals=vals.Multiples(min_value=0, max_value=8000000, divisor=8))

        self.add_parameter(
            'burst_count',
            get_cmd='TRIGGER:COUNT?',
            set_cmd='TRIGGER:COUNT: {}',
            get_parser=float,
            vals=vals.Ints(1, 1048576),
            docstring='Perform number of waveform cycles after trigger.')


        # Waveform parameters
        self.add_parameter(
            'uploaded_waveforms',
            parameter_class=ManualParameter,
            initial_value=[],
            vals=vals.Lists(),
            docstring='List of uploaded waveforms')

        self.add_parameter(
            'waveform_timing',
            get_cmd='TRACE:SELECT:TIMING?',
            set_cmd='TRACE:SELECT:TIMING {}',
            vals=EnumVisa('COHerent', 'IMMediate'),
            docstring='Possible waveform timings are: ' 
                      'coherent: finish current waveform before continuing. '
                      'immediate: immediately skip to next waveform')


        # Sequence parameters
        self.add_parameter(
            'uploaded_sequence',
            parameter_class=ManualParameter,
            vals=vals.Iterables())

        self.add_parameter(
            'sequence_mode',
            get_cmd='SEQUENCE:ADVANCE?',
            set_cmd='SEQUENCE:ADVANCE {}',
            vals=EnumVisa('AUTOmatic', 'ONCE', 'STEPped'),
            docstring='Possible sequence modes are: '
                      'automatic: Automatically continue to next waveform, '
                      'and repeat sequence from start when finished. When '
                      'encountering jump command, wait for event input '
                      'before continuing. '
                      'once: Automatically continue to next waveform. Stop '
                      'when sequence completed. '
                      'stepped: wait for event input after each waveform. '
                      'Repeat sequence when completed.')

        self.add_parameter(
            'sequence_jump',
            get_cmd='SEQUENCE:JUMP?',
            set_cmd='SEQUENCE:JUMP {}',
            vals=EnumVisa('BUS', 'EVENt'),
            docstring='Determines the trigger source that will cause the '
                      'sequence to advance after a jump bit. '
                      'Possible jump modes are: '
                      'bus: only advance after a remote trigger command. '
                      'event: only advance after an event input.')

        self.add_parameter(
            'sequence_select_source',
            get_cmd='SEQUENCE:SELECT:SOURCE?',
            set_cmd='SEUQENCE:SELECT:SOURCE {}',
            vals=EnumVisa('BUS', 'EXTernal'),
            docstring='Possible sources that can select active sequence: '
                      'bus: sequence switches when remote command is called. '
                      'external: rear panel connector can dynamically choose '
                      'next sequence.')

        self.add_parameter(
            'sequence_select_timing',
            get_cmd='SEQUENCE:SELECT:TIMING?',
            set_cmd='SEQUENCE:SELECT:TIMING {}',
            vals=EnumVisa('COHerent', 'IMMediate'),
            docstring='Possible ways in which the generator transitions from '
                      'sequence to sequence: '
                      'coherent: transition once current sequence is done. '
                      'immediate: abort current sequence and progress to next')


        # Functions
        self.add_function(
            'enable',
            call_cmd=f'INST {self.id};ENABLE',
            docstring='An immediate and unconditional generation of the '
                      'selected output waveform. Must be armed and in '
                      'continuous mode.')
        self.add_function(
            'abort',
            call_cmd=f'INST {self.id};ABORT',
            docstring='An immediate and unconditional termination of the '
                      'output waveform.')

        self.add_function(
            'trigger',
            call_cmd=f'INST {self.id};TRIGGER',
            docstring='Perform software trigger')

    def add_parameter(self, name, parameter_class=ChannelVisaParameter,
                      **kwargs):
        # Override add_parameter such that it uses ChannelVisaParameter
        if parameter_class == ChannelVisaParameter:
            kwargs['channel_id'] = self.id

        super().add_parameter(name,
                              parameter_class=parameter_class,
                              **kwargs)

    def add_waveform(self, waveform, segment_number=None):
        if segment_number is None:
            segment_number = len(self.uploaded_waveforms()) + 1

        assert len(waveform) >= 320, 'Waveform must have at least 320 points'
        assert not len(waveform) % 32, 'Waveform points must be divisible by 32'
        assert segment_number <= len(self.uploaded_waveforms()) + 1, \
            "segment number is larger than number of uploaded waveforms + 1"

        # Set active channel to current channel if necessary
        if self._parent.active_channel.get_latest() != self.id:
            self._parent.active_channel(self.id)

        self.write(f'TRACE:DELETE {segment_number}')
        self.write(f'TRACE:DEFINE {segment_number},{len(waveform)}')
        self.write(f'TRACE:SELECT {segment_number}')

        # Waveform points are 12 bits, which are converted to 2 bytes
        number_of_bytes = len(waveform) * 2
        waveform_DAC = (2**13 - 1) + (2**12 - 1) * waveform
        waveform_DAC = waveform_DAC.astype('int')

        # Add stop bytes (set 15th bit of last 32 words to 1)
        waveform_DAC[-32:] += 2**14

        # Convert waveform to bytes (note that Endianness is important,
        # might not work on Mac/Linux)
        waveform_bytes = array.array('H', waveform_DAC).tobytes()

        # Send header, format is #{digits in number of bytes}{number of bytes}
        # followed by the binary waveform data.
        # The hash denotes that binary data follows.
        # use write_raw to avoid termination/encoding
        self.visa_handle.write_raw(
            f'TRACE#{len(str(number_of_bytes))}{number_of_bytes}')

        # Send waveform as binary data
        # If this stage fails the instrument can freeze and need a power cycle
        return_bytes, _ = self.visa_handle.write_raw(waveform_bytes)

        if return_bytes != len(waveform_bytes):
            warnings.warn(f'Unsuccessful waveform transmission. Transmitted '
                          f'{return_bytes} instead of {len(waveform_bytes)}')

        # Add waveform to parameter
        if segment_number - 1 < len(self.uploaded_waveforms()):
            self.uploaded_waveforms()[segment_number - 1] = waveform
        else:
            self.uploaded_waveforms().append(waveform)

        return waveform_DAC


    def clear_waveforms(self):
        # Set active channel to current channel if necessary
        if self._parent.active_channel.get_latest() != self.id:
            self._parent.active_channel(self.id)

        self.write('TRACE:DELETE:ALL')
        self.uploaded_waveforms([])

    def set_sequence(self, sequence, id=1):
        """
        Set sequence of a channel, clearing its previous sequence.
        All instructions in the sequence must have the following 3 elements:
            (segment_number, loops, jump_flag)
                segment_number is the waveform index
                loops specifies how many times the waveform is repeated.
                    Note that for stepped mode, the waveform is repeated
                    without needing intermediate triggers
                jump_flag specifies if it should wait for a valid input
                    event before continuing. This is ignored in stepped mode.

        Args:
            sequence: list of 3-element instructions
            id (int): sequence id (1 by default)
        Returns:
            None
        """
        assert all(len(instruction) == 3 for instruction in sequence), \
            "All instructions must be of form: " \
            "(segment_number, loops, jump_flag)."
        assert 2 < len(sequence) <= 32768 , \
            "Must have between 3 and 32768 instructions"
        assert all(0 < instruction[0] <= 32000 for instruction in sequence), \
            "Segment numbers must be between 1 and 32000"
        assert all(0 < instruction[1] <= 1048575 for instruction in sequence),\
            "Number of loops must be between 1 and 1,048,575"
        assert all(instruction[2] in [0, 1] for instruction in sequence), \
            "Jump flag must be either 0 or 1"
        assert 0 < id <= 1000, "Sequence id must be between 1 and 1000"

        # Set active channel to current channel if necessary
        if self._parent.active_channel.get_latest() != self.id:
            self._parent.active_channel(self.id)

        run_mode = self.run_mode()
        if run_mode == 'sequenced':
            # Temporarily change run mode, as the sequence is restarted
            # after every SEQ:DEFINE command.
            self.run_mode('user')

        # Delete any previous sequence
        self.write('SEQUENCE:DELETE:ALL')
        # Select sequence id
        self.write(f'SEQUENCE SELECT {id}')
        # Specify length of sequence
        self.write(f'SEQUENCE:LENGTH {len(sequence)}')

        # TODO program sequence via binary code. This speeds up programming
        for step, (segment_number, loop, jump) in enumerate(sequence):
            step += 1 # Step is 1-based

            self.write(f'SEQ:DEFINE {step},{segment_number},{loop},{jump}')

        if run_mode == 'sequenced':
            # Restore sequenced run mode
            self.run_mode(run_mode)

        self.uploaded_sequence(sequence)

class Keysight_81180A(VisaInstrument):
    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, **kwargs)

        self.visa_handle.read_termination = '\n'

        # Main parameters
        self.add_parameter(
            'active_channel',
            get_cmd='INSTRUMENT?',
            set_cmd='INSTRUMENT {}',
            get_parser=int,
            docstring='Select channel for programming. Note that nearly all '
                      'parameters are channel-dependent. Exceptions are LAN '
                      'configuration commands, Store/recall commands, '
                      'System Commands and common commands')

        self.add_parameter(
            'get_error',
            get_cmd='SYSTEM:ERROR?',
            get_parser=str.rstrip,
            docstring='Get oldest error.')

        self.add_function(
            'reset',
            call_cmd='*RST')

        # Add channels containing their own parameters/functions
        for channel_id in [1, 2]:
            channel_name=f'ch{channel_id}'
            channel = AWGChannel(self, name=channel_name, id=channel_id)
            setattr(self, channel_name, channel)

        self.channels = ChannelList(parent=self,
                                    name='channels',
                                    chan_type=AWGChannel,
                                    chan_list=[self.ch1, self.ch2])
        self.add_submodule('channels', self.channels)
