import array
from warnings import warn
from time import sleep, time
import logging
import visa
import struct

from qcodes import (
    VisaInstrument,
    InstrumentChannel,
    ChannelList,
    Parameter,
    ManualParameter,
)
from qcodes import validators as vals
from qcodes.utils.validators import EnumVisa
from qcodes.utils.helpers import arreqclose_in_list

logger = logging.getLogger(__name__)


"""The Keysight 81180A AWG has three main run modes, set via AWG.ch.run_mode():

"Continuous run" mode repeats a waveform indefinitely without requiring an external
trigger. This mode has two submodes, "self" 

"""


class ChannelVisaParameter(Parameter):
    """Parameter wrapper that ensures active_channel is set to current channel"""

    def __init__(
        self, name, channel_id, vals=None, val_mapping=None, **kwargs
    ):
        if isinstance(vals, EnumVisa) and val_mapping is None:
            val_mapping = vals.val_mapping

        super().__init__(
            name, vals=vals, val_mapping=val_mapping, **kwargs
        )
        self.channel_id = channel_id

        self.get_raw = self._wrap_channel_get(self.get_raw)
        self.get = self._wrap_get(self.get_raw)
        self.set_raw = self._wrap_channel_set(self.set_raw)
        self.set = self._wrap_set(self.set_raw)

    def ensure_correct_channel(self):
        """Set active channel to current channel if necessary.
        Sometimes a change channel command is not received, and so we explicitly
        check afterwards
        """
        active_channel = self._instrument._parent.active_channel.get_latest()
        if active_channel != self.channel_id:
            for attempt in range(10):
                self._instrument._parent.active_channel(self.channel_id)
                sleep(0.01)
                if self._instrument._parent.active_channel() == self.channel_id:
                    break
            else:
                warn(f"Could not set the 81180 active channel to {self.channel_id}")

    def _wrap_channel_get(self, get_raw):
        def channel_get_wrapper():
            self.ensure_correct_channel()
            return get_raw()

        return channel_get_wrapper

    def _wrap_channel_set(self, set_raw):
        def channel_set_wrapper(*args, **kwargs):
            self.ensure_correct_channel()
            return set_raw(*args, **kwargs)

        return channel_set_wrapper


class AWGChannel(InstrumentChannel):
    def __init__(self, parent, name, id, **kwargs):
        super().__init__(parent, name, **kwargs)

        self.id = id

        self.write = self.parent.write
        self.visa_handle = self.parent.visa_handle

        self.add_parameter(
            "continuous_run_mode",
            get_cmd="INITIATE:CONTINUOUS?",
            set_cmd="INITIATE:CONTINUOUS {}",
            val_mapping={True: "ON", False: "OFF"},
            docstring="Choose to run in continuous mode (1) or triggered and gated mode (0)"
            "Continuous run mode repeats a waveform indefinitely without"
            "requiring a trigger.",
        )

        self.add_parameter(
            "continuous_armed",
            get_cmd="INITIATE:CONTINUOUS:ENABLE?",
            set_cmd="INITIATE:CONTINUOUS:ENABLE {}",
            val_mapping={True: "ARM", False: "SELF"},
            docstring="Whether continuous mode should wait to be armed:\n"
            "\tWhen False, waveforms are straight away generated as "
            "soon as the output is on. "
            "\tWhen True, the AWG remains idle until an enable signal "
            "is received. During idling, the output depends on the "
            "output function.",
        )

        self.add_parameter(
            "sample_rate",
            get_cmd="FREQUENCY:RASTER?",
            set_cmd="FREQUENCY:RASTER {:.8f}",
            get_parser=float,
            vals=vals.Numbers(10e6, 4.2e9),
            docstring="Set the sample rate of the arbitrary waveforms. Has no "
            "effect on standard waveforms",
        )

        self.add_parameter(
            "frequency_standard_waveforms",
            unit="Hz",
            get_cmd="FREQUENCY?",
            set_cmd="FREQUENCY {:.8f}",
            get_parser=float,
            vals=vals.Numbers(10e-3, 250e6),
            docstring="Set the frequency (Hz) of standard waveforms",
        )

        self.add_parameter(
            "function_mode",
            get_cmd="FUNCTION:MODE?",
            set_cmd="FUNCTION:MODE {}",
            vals=EnumVisa(
                "FIXed", "USER", "SEQuenced", "ASEQuenced", "MODulated", "PULSe"
            ),
            docstring="Run mode defines the type of waveform that is "
            "available. Possible run modes are:\n"
            "\tfixed: standard waveform shapes.\n"
            "\tuser: arbitrary waveform shapes.\n"
            "\tsequenced: sequence of arbitrary waveforms.\n"
            "\tasequenced: advanced sequence of arbitrary "
            "waveforms.\n"
            "\tmodulated: modulated (standard) waveforms.\n"
            "\tpulse: digital pulse function",
        )

        self.add_parameter(
            "output_coupling",
            get_cmd="OUTPUT:COUPLING?",
            set_cmd="OUTPUT:COUPLING {}",
            vals=vals.Enum("DC", "DAC", "AC"),
            docstring="Possible output couplings are:\n"
            "\tDC: optimized for pulses at high amplitudes.\n"
            "\tDAC: optimized for bandwidth but low amplitude.\n"
            "\tAC: optimized for bandwidth.\n"
            "Note that DC and DAC use the DC output, and AC uses "
            "the AC output. DC and DAC couplings also allow "
            "control of amplitude and offset",
        )

        self.add_parameter(
            "output",
            get_cmd="OUTPUT?",
            set_cmd="OUTPUT {}",
            vals=EnumVisa("ON", "OFF"),
        )

        self.add_parameter(
            "sync",
            get_cmd="OUTPUT:SYNC?",
            set_cmd="OUTPUT:SYNC {}",
            vals=EnumVisa("ON", "OFF"),
        )

        self.add_parameter(
            "power",
            unit="dBm",
            get_cmd="POWER?",
            set_cmd="POWER {:.8f}",
            get_parser=float,
            vals=vals.Numbers(-5, 5),  # Might be -5 to 5, manual is unclear
            docstring="Set output power (dBm) from AC output path (50-ohm matched)",
        )

        self.add_parameter(
            "voltage_DAC",
            unit="V",
            get_cmd="VOLTAGE:DAC?",
            set_cmd="VOLTAGE:DAC {:.8f}",
            get_parser=float,
            vals=vals.Numbers(50e-3, 2),
            docstring="Waveform amplitude (V) when routed through the DAC path",
        )

        self.add_parameter(
            "voltage_DC",
            unit="V",
            get_cmd="VOLTAGE?",
            set_cmd="VOLTAGE {:.8f}",
            get_parser=float,
            vals=vals.Numbers(50e-3, 2),
            docstring="Waveform amplitude (V) when routed through the DC path",
        )

        self.add_parameter(
            "voltage_offset",
            unit="V",
            get_cmd="VOLTAGE:OFFSET?",
            set_cmd="VOLTAGE:OFFSET {:.8f}",
            get_parser=float,
            vals=vals.Numbers(-1.5, 1.5),
            docstring="Voltage offset (V) when routed through DAC or DC path",
        )

        self.add_parameter(
            "output_modulation",
            get_cmd="MODulation:TYPE?",
            set_cmd="MODulation:TYPE {}",
            vals=EnumVisa("OFF", "AM", "FM", "SWEEP"),
        )

        # Trigger parameters
        self.add_parameter(
            "trigger_input",
            set_cmd="TRIGGER:{}",
            vals=vals.Enum("ECL", "TTL"),
            docstring="Set trigger input. Possible inputs are:\n"
            "\tECL: negative signal at fixed -1.3V.\n"
            "\tTTL: variable trigger_level",
        )

        self.add_parameter(
            "trigger_source",
            get_cmd="TRIGGER:SOURCE:ADVANCE?",
            set_cmd="TRIGGER:SOURCE:ADVANCE {}",
            vals=EnumVisa("EXTernal", "BUS", "TIMer", "EVENt"),
            docstring="Possible trigger sources are:\n"
            "\texternal: Use TRIG IN port exclusively.\n"
            "\tbus: Use remote commands exclusively.\n"
            "\ttimer: Use internal trigger generator exclusively.\n"
            "\tevent: Use EVENT IN port exclusively.",
        )

        self.add_parameter(
            "trigger_mode",
            get_cmd="TRIGGER:MODE?",
            set_cmd="TRIGGER:MODE {}",
            vals=EnumVisa("NORMal", "OVERride"),
            docstring="Possible trigger modes are:\n"
            "\tnormal:  the first trigger activates the output and "
            "consecutive triggers are ignored for the duration of "
            "the output waveform.\n"
            "\toverride:  the first trigger activates  the output "
            "and consecutive triggers  restart the output waveform, "
            "regardless if the current waveform has been completed "
            "or not.",
        )

        self.add_parameter(
            "trigger_timer_mode",
            get_cmd="TRIGGER:TIMER:MODE?",
            set_cmd="TRIGGER:TIMER:MODE {}",
            vals=EnumVisa("TIME", "DELay"),
            docstring="Possible modes of internal trigger are:\n"
            "\ttime: Perform trigger at fixed time interval.\n"
            "\tdelay: Perform trigger at fixed time intervals after "
            "previous waveform finished.",
        )

        self.add_parameter(
            "trigger_timer_time",
            get_cmd="TRIGGER:TIMER:TIME?",
            set_cmd="TRIGGER:TIMER:TIME {:f}",
            get_parser=float,
            unit="s",
            vals=vals.Numbers(100e-9, 20),
            docstring="Triggering period (s) when in internal trigger timer mode"
        )

        self.add_parameter(
            "trigger_timer_delay",
            get_cmd="TRIGGER:TIMER:DELAY?",
            set_cmd="TRIGGER:TIMER:DELAY {:d}",
            get_parser=int,
            unit="1/sample rate",
            vals=vals.Multiples(min_value=152, max_value=8000000, divisor=8),
            docstring="Delay of the internal trigger generator."
        )

        self.add_parameter(
            "trigger_level",
            get_cmd="TRIGGER:LEVEL?",
            set_cmd="TRIGGER:LEVEL {:.4f}",
            get_parser=float,
            vals=vals.Numbers(-5, 5),
            unit="V",
            docstring="Trigger level when trigger_mode is TTL",
        )

        self.add_parameter(
            "trigger_slope",
            get_cmd="TRIGGER:SLOPE?",
            set_cmd="TRIGGER:SLOPE {}",
            vals=EnumVisa("POSitive", "NEGative", "EITher"),
        )

        self.add_parameter(
            "trigger_delay",
            get_cmd="TRIGGER:DELAY?",
            set_cmd="TRIGGER:DELAY {:.4f}",
            get_parser=float,
            unit="1/sample rate",
            vals=vals.Multiples(min_value=0, max_value=8000000, divisor=8),
            docstring="Delay between receiving an external trigger, and "
                      "outputting the first waveform"
        )

        self.add_parameter(
            "burst_count",
            get_cmd="TRIGGER:COUNT?",
            set_cmd="TRIGGER:COUNT {}",
            set_parser=int,
            get_parser=int,
            vals=vals.Numbers(1, 1048576),
            docstring="Perform number of waveform cycles after trigger.",
        )

        # Waveform parameters
        self.add_parameter(
            "uploaded_waveforms",
            set_cmd=None,
            initial_value=[],
            vals=vals.Lists(),
            docstring="List of uploaded waveforms",
        )

        self.add_parameter(
            "waveform_timing",
            get_cmd="TRACE:SELECT:TIMING?",
            set_cmd="TRACE:SELECT:TIMING {}",
            vals=EnumVisa("COHerent", "IMMediate"),
            docstring="Possible waveform timings are:\n"
            "\tcoherent: finish current waveform before next.\n"
            "\timmediate: immediately skip to next waveform",
        )

        # Sequence parameters
        self.add_parameter(
            "uploaded_sequence", parameter_class=ManualParameter, vals=vals.Iterables()
        )

        self.add_parameter(
            "sequence_mode",
            get_cmd="SEQUENCE:ADVANCE?",
            set_cmd="SEQUENCE:ADVANCE {}",
            vals=EnumVisa("AUTOmatic", "ONCE", "STEPped"),
            docstring="Possible sequence modes are:\n"
            "\tautomatic: Automatically continue to next waveform, "
            "and repeat sequence from start when finished. When "
            "encountering jump command, wait for event input "
            "before continuing.\n"
            "\tonce: Automatically continue to next waveform. Stop "
            "when sequence completed.\n"
            "\tstepped: wait for event input after each waveform. "
            "Repeat sequence when completed.",
        )

        self.add_parameter(
            "sequence_once_count",
            get_cmd="SEQUENCE:ONCE:COUNT?",
            set_cmd="SEQUENCE:ONCE:COUNT {}",
            set_parser=int,
            get_parser=int,
            vals=vals.Numbers(1, 1048575),
            docstring="Determines the number of times a waveform sequence will"
            "be repeated.",
        )

        self.add_parameter(
            "sequence_jump",
            get_cmd="SEQUENCE:JUMP?",
            set_cmd="SEQUENCE:JUMP {}",
            vals=EnumVisa("BUS", "EVENt"),
            docstring="Determines the trigger source that will cause the "
            "sequence to advance after a jump bit. "
            "Possible jump modes are:\n"
            "\tbus: only advance after a remote trigger command.\n"
            "\tevent: only advance after an event input.",
        )

        self.add_parameter(
            "sequence_select_source",
            get_cmd="SEQUENCE:SELECT:SOURCE?",
            set_cmd="SEUQENCE:SELECT:SOURCE {}",
            vals=EnumVisa("BUS", "EXTernal"),
            docstring="Possible sources that can select active sequence:\n"
            "\tbus: sequence switches when remote command is "
            "called.\n"
            "\texternal: rear panel connector can dynamically "
            "choose "
            "next sequence.",
        )

        self.add_parameter(
            "sequence_select_timing",
            get_cmd="SEQUENCE:SELECT:TIMING?",
            set_cmd="SEQUENCE:SELECT:TIMING {}",
            vals=EnumVisa("COHerent", "IMMediate"),
            docstring="Possible ways in which the generator transitions from "
            "sequence to sequence:\n"
            "c\toherent: transition once current sequence is done.\n"
            "\timmediate: abort current sequence and progress to "
            "next",
        )

        self.add_parameter(
            "sequence_pre_step",
            get_cmd="SEQUENCE:PREStep?",
            set_cmd="SEQUENCE:PREStep {}",
            vals=EnumVisa("WAVE", "DC"),
            docstring="Choose to play a blank DC segment while waiting for an "
            "event signal to initiate or continue a sequence.",
        )

        # Functions
        self.add_function(
            "enable",
            call_cmd=f"INST {self.id};ENABLE",
            docstring="An immediate and unconditional generation of the "
            "selected output waveform. Must be armed and in "
            "continuous mode.",
        )
        self.add_function(
            "abort",
            call_cmd=f"INST {self.id};ABORT",
            docstring="An immediate and unconditional termination of the "
            "output waveform.",
        )

        self.add_function(
            "trigger",
            call_cmd=f"INST {self.id};TRIGGER",
            docstring="Perform software trigger",
        )

        self.add_function(
            "on", call_cmd=f"INST {self.id};OUTPUT ON", docstring="Turn output on"
        )

        self.add_function(
            "off", call_cmd=f"INST {self.id};OUTPUT OFF", docstring="Turn output off"
        )

        # Query current values for all parameters
        for param_name, param in self.parameters.items():
            param()

    def add_parameter(self, name, parameter_class=ChannelVisaParameter, **kwargs):
        # Override add_parameter such that it uses ChannelVisaParameter
        if parameter_class == ChannelVisaParameter:
            kwargs["channel_id"] = self.id

        super().add_parameter(name, parameter_class=parameter_class, **kwargs)

    def upload_waveform(self, waveform, segment_number=None):
        if segment_number is None:
            segment_number = len(self.uploaded_waveforms()) + 1

        assert len(waveform) >= 320, "Waveform must have at least 320 points"
        assert not len(waveform) % 32, "Waveform points must be divisible by 32"
        assert (
            segment_number <= len(self.uploaded_waveforms()) + 1
        ), "segment number is larger than number of uploaded waveforms + 1"
        if self.output_coupling.get_latest() == "DAC":
            assert (
                max(abs(waveform)) <= 0.5
            ), "In DAC run mode, waveform voltage cannot exceed +-0.5V"
        else:
            assert (
                max(abs(waveform)) <= 2
            ), "In DC/AC run mode, waveform voltage cannot exceed +-2V"

        # Set active channel to current channel if necessary
        if self._parent.active_channel.get_latest() != self.id:
            self._parent.active_channel(self.id)

        if segment_number - 1 < len(self.uploaded_waveforms()):
            # Waveform already exists
            # TODO check that new waveform does not have more points than previous
            self.write(f"TRACE:DELETE {segment_number}")

        self.write(f"TRACE:DEFINE {segment_number},{len(waveform)}")
        self.write(f"TRACE:SELECT {segment_number}")

        # Waveform points are 12 bits, which are converted to 2 bytes
        number_of_bytes = len(waveform) * 2
        # Convert waveform to 12bit (min = 0, max = 2**12 - 1)
        if self.output_coupling.get_latest() == "DAC":
            # Voltage limits are +- 0.5V.
            waveform_12bit = (2 ** 11 - 1) + (2 ** 12 - 1) * waveform
        else:
            waveform_12bit = (2 ** 11 - 1) + (2 ** 10 - 1) * waveform
        waveform_12bit = waveform_12bit.astype("int")

        # Add stop bytes (set 15th bit of last 32 words to 1)
        waveform_12bit[-32:] += 2 ** 14

        # Convert waveform to bytes (note that Endianness is important,
        # might not work on Mac/Linux)
        waveform_bytes = array.array("H", waveform_12bit).tobytes()

        # Send header, format is #{digits in number of bytes}{number of bytes}
        # followed by the binary waveform data.
        # The hash denotes that binary data follows.
        # use write_raw to avoid termination/encoding
        self.visa_handle.write_raw(
            f"TRACE#{len(str(number_of_bytes))}{number_of_bytes}"
        )

        # Send waveform as binary data
        # If this stage fails the instrument can freeze and need a power cycle
        return_bytes, _ = self.visa_handle.write_raw(waveform_bytes)

        if return_bytes != len(waveform_bytes):
            warn(
                f"Unsuccessful waveform transmission. Transmitted "
                f"{return_bytes} instead of {len(waveform_bytes)}"
            )

        # Add waveform to parameter
        if segment_number - 1 < len(self.uploaded_waveforms()):
            self.uploaded_waveforms()[segment_number - 1] = waveform
        else:
            self.uploaded_waveforms().append(waveform)

        return segment_number

    def upload_waveforms(self, waveforms, append=False, allow_existing=False):
        """Uploade a list of waveforms


        TODO:
            allow_existing can be improved by first checking which new waveforms
            already exist, and then subsequently checking if the maximum size
            is exceeded
        """
        total_new_waveform_points = sum(map(len, waveforms))
        if total_new_waveform_points > self._parent.waveform_max_length:
            raise RuntimeError(
                f'Total waveform points {total_new_waveform_points} exceeds '
                f'limit of 81180A ({self._parent.waveform_max_length})'
            )

        total_existing_waveform_points = sum(map(len, self.uploaded_waveforms()))
        total_waveform_points = total_existing_waveform_points + total_new_waveform_points

        waveform_idx_mapping = {}

        if append:  # Append all new waveforms to existing waveforms
            if total_waveform_points > self._parent.waveform_max_length:
                raise RuntimeError(
                    f'Total existing and new waveform points combined '
                    f'{total_new_waveform_points} exceeds limit of 81180A '
                    f'({self._parent.waveform_max_length}). Please set force=False'
                )
            else:
                for k, waveform in enumerate(waveforms, start=1):
                    idx = len(self.uploaded_waveforms()) + 1  # 1-based indexing
                    waveform_idx_mapping[k] = idx
                    self.upload_waveform(waveform, idx)
        elif allow_existing:  # Only append new waveforms if not already existing
            if total_waveform_points > self._parent.waveform_max_length:
                self.clear_waveforms()
            for k, waveform in enumerate(waveforms, start=1):  # 1-based indexing
                idx = arreqclose_in_list(waveform, self.uploaded_waveforms(), atol=1e-3)
                if idx is None:
                    idx = self.upload_waveform(waveform)
                else:
                    idx += 1  # 1-based indexing
                waveform_idx_mapping[k] = idx
        else:
            for k, waveform in enumerate(waveforms, start=1):  # 1-based indexing
                self.upload_waveform(waveform, k)
                waveform_idx_mapping[k] = k

        return waveform_idx_mapping

    def clear_waveforms(self):
        # Set active channel to current channel if necessary
        if self._parent.active_channel.get_latest() != self.id:
            self._parent.active_channel(self.id)

        self.write("TRACE:DELETE:ALL")
        self.uploaded_waveforms([])

    def set_sequence(self, sequence, id: int = 1, binary: bool = True):
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
            A fourth element can be included (the segment label), which is
            ignored and only used for debugging purposes

        Args:
            sequence: list of 3-element instructions
            id: sequence id (1 by default)
            bineray: Whether to upload sequence as binary data (faster)
        Returns:
            None
        """
        assert all(len(instruction) in [3, 4] for instruction in sequence), (
            "All instructions must be of form: " "(segment_number, loops, jump_flag)."
        )
        assert 2 < len(sequence) <= 32768, "Must have between 3 and 32768 instructions"
        assert all(
            0 < instruction[0] <= 32000 for instruction in sequence
        ), f"Segment numbers must be between 1 and 32000, sequence: {sequence}"
        assert all(
            0 < instruction[1] <= 1048575 for instruction in sequence
        ), f"Number of loops must be between 1 and 1,048,575, sequence: {sequence}"
        assert all(
            instruction[2] in [0, 1] for instruction in sequence
        ), f"Jump flag must be either 0 or 1, sequence: {sequence}"
        assert 0 < id <= 1000, f"Sequence id must be between 1 and 1000, not {id}"

        # Set active channel to current channel if necessary
        if self._parent.active_channel.get_latest() != self.id:
            self._parent.active_channel(self.id)

        # Delete any previous sequence
        self.write("SEQUENCE:DELETE:ALL")
        # Select sequence id
        self.write(f"SEQUENCE:SELECT {id}")
        # Specify length of sequence
        self.write(f"SEQUENCE:LENGTH {len(sequence)}")

        if binary:
            num_bytes = 8 * len(sequence)  # Each sequence has eight bytes
            self.visa_handle.write_raw(f"SEQUENCE#{len(str(num_bytes))}{num_bytes}")

            sequence_bytes = b''.join(
                struct.pack('IHH', instr[1], instr[0], instr[2]) for instr in sequence
            )

            # Send waveform as binary data
            # If this stage fails the instrument can freeze and need a power cycle
            return_bytes, _ = self.visa_handle.write_raw(sequence_bytes)

            if return_bytes != len(sequence_bytes):
                warn(
                    f"Unsuccessful sequence transmission. Transmitted "
                    f"{return_bytes} instead of {len(sequence_bytes)}"
                )
        else:
            function_mode = self.function_mode()
            if function_mode == "sequenced":
                # Temporarily change run mode, as the sequence is restarted
                # after every SEQ:DEFINE command.
                self.function_mode("user")

            # TODO program sequence via binary code. This speeds up programming
            for step, (segment_number, loop, jump, *label) in enumerate(sequence):
                step += 1  # Step is 1-based
                self.write(f"SEQ:DEFINE {step},{segment_number},{loop},{jump}")

            if function_mode == "sequenced":
                # Restore sequenced run mode
                self.function_mode(function_mode)

        self.uploaded_sequence(sequence)


class Keysight_81180A(VisaInstrument):
    waveform_max_length = 16000000  # Verified that limit is precisely 16M per channel
    _operation_complete_timing = {}

    def __init__(self, name, address, debug_mode=False, **kwargs):
        super().__init__(name, address, **kwargs)

        self.ensure_idle = False
        self.debug_mode = debug_mode

        self.set_terminator("\n")

        # Main parameters
        self.add_parameter(
            "active_channel",
            get_cmd="INSTRUMENT?",
            set_cmd="INSTRUMENT {}",
            get_parser=int,
            docstring="Select channel for programming, automatically set by all"
            "parameters belonging to an AWG channel. Note that nearly "
            "all parameters are channel-dependent. Exceptions are LAN "
            "configuration commands, Store/recall commands, "
            "System Commands and common commands",
        )

        self.add_parameter(
            "get_error",
            get_cmd="SYSTEM:ERROR?",
            get_parser=str.rstrip,
            docstring="Get oldest error.",
        )

        self.add_parameter(
            "couple_state",
            get_cmd="INSTrument:COUPle:STATe?",
            set_cmd="INSTrument:COUPle:STATe {}",
            vals=EnumVisa("OFF", "ON"),
            docstring="Sets or queries the couple state of the synchronized channels. "
            "When enabled, the sample clock of channel 1 will feed the "
            "channel 2 and the start phase of the channel 2 channels "
            "will lock to the channel 1 waveform.",
        )

        self.add_parameter(
            "channel_skew",
            get_cmd="INSTrument:COUPle:SKEW?",
            set_cmd="INSTrument:COUPle:SKEW {}",
            get_parser=float,
            vals=vals.Numbers(-3e-9, 3e-9),
            docstring="When couple state is ON, this command sets or queries "
            "the skew between the two channels. Skew defines fine "
            "offset between channels in units of time. Only channel 2 "
            "has the skew is computed in reference to channel 1.",
        )

        self.add_parameter(
            "is_idle", get_cmd="*OPC?",
        )

        self.add_function(
            "reset",
            call_cmd="*RST",
            docstring='Perform an instrument reset. Note that this briefly '
                      'outputs a voltage exceeding 1V.'
        )

        self.add_function(
            "clear_errors", call_cmd="*CLS", docstring="Clear all errors."
        )

        # Query current values for all parameters
        for param_name, param in self.parameters.items():
            param()

        # Add channels containing their own parameters/functions
        for channel_id in [1, 2]:
            channel_name = f"ch{channel_id}"
            channel = AWGChannel(self, name=channel_name, id=channel_id)
            setattr(self, channel_name, channel)

        self.channels = ChannelList(
            parent=self,
            name="channels",
            chan_type=AWGChannel,
            chan_list=[self.ch1, self.ch2],
        )
        self.add_submodule("channels", self.channels)


    def write(self, cmd: str) -> None:
        super().write(cmd)

        # Verify that the operation has been completed successfully
        # Also record time taken to request operation complete
        if self.ensure_idle and cmd != '*CLS':
            t0 = time()
            idle_result = self.is_idle()
            duration = time() - t0
            if not idle_result == "1":
                logger.warning(f"Operation is not idle after command {cmd}, result {idle_result}")
                sleep(1)
                try:
                    self.visa_handle.read()
                except visa.VisaIOError:
                    logger.warning(f"Additional visa.read produced timeout error")

            key = cmd.split(" ")[0]
            val = self._operation_complete_timing.get(key, [0, 0])
            val[0] += duration  # Add time of current operation
            val[1] += 1  # Increment counter by 1
            self._operation_complete_timing[key] = val

        if self.debug_mode:
            error = self.get_error()
            if error != "0,No error":
                logger.warning(f"Command {cmd} has caused error message: {error}")
                sleep(1)
                try:
                    self.visa_handle.read()
                except visa.VisaIOError:
                    logger.warning(f"Additional visa.read produced timeout error")

                self.clear_errors()

    def average_operation_complete_timings(self):
        return {
            key: val[0] / val[1] for key, val in self._operation_complete_timing.items()
        }