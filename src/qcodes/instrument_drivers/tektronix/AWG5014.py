import array as arr
import logging
import re
import struct
from collections import abc
from collections.abc import Sequence
from io import BytesIO
from time import localtime, sleep
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    NamedTuple,
    Optional,
    Union,
    cast,
)

import numpy as np
from pyvisa.errors import VisaIOError

from qcodes import validators as vals
from qcodes.instrument import VisaInstrument, VisaInstrumentKWArgs

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from qcodes.parameters import Parameter

log = logging.getLogger(__name__)


class _MarkerDescriptor(NamedTuple):
    marker: int
    channel: int


def parsestr(v: str) -> str:
    return v.strip().strip('"')


class TektronixAWG5014(VisaInstrument):
    """
    This is the QCoDeS driver for the Tektronix AWG5014
    Arbitrary Waveform Generator.

    The driver makes some assumptions on the settings of the instrument:

        - The output channels are always in Amplitude/Offset mode
        - The output markers are always in High/Low mode

    Todo:
        - Implement support for cable transfer function compensation
        - Implement more instrument functionality in the driver
        - Remove double functionality
        - Remove inconsistensies between the name of a parameter and
          the name of the same variable in the tektronix manual

    In the future, we should consider the following:

        * Removing test_send??
        * That sequence element (SQEL) parameter functions exist but no
          corresponding parameters.

    """

    AWG_FILE_FORMAT_HEAD: ClassVar[dict[str, str]] = {
        'SAMPLING_RATE': 'd',    # d
        'REPETITION_RATE': 'd',    # # NAME?
        'HOLD_REPETITION_RATE': 'h',    # True | False
        'CLOCK_SOURCE': 'h',    # Internal | External
        'REFERENCE_SOURCE': 'h',    # Internal | External
        'EXTERNAL_REFERENCE_TYPE': 'h',    # Fixed | Variable
        'REFERENCE_CLOCK_FREQUENCY_SELECTION': 'h',
        'REFERENCE_MULTIPLIER_RATE': 'h',    #
        'DIVIDER_RATE': 'h',   # 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256
        'TRIGGER_SOURCE': 'h',    # Internal | External
        'INTERNAL_TRIGGER_RATE': 'd',    #
        'TRIGGER_INPUT_IMPEDANCE': 'h',    # 50 ohm | 1 kohm
        'TRIGGER_INPUT_SLOPE': 'h',    # Positive | Negative
        'TRIGGER_INPUT_POLARITY': 'h',    # Positive | Negative
        'TRIGGER_INPUT_THRESHOLD': 'd',    #
        'EVENT_INPUT_IMPEDANCE': 'h',    # 50 ohm | 1 kohm
        'EVENT_INPUT_POLARITY': 'h',    # Positive | Negative
        'EVENT_INPUT_THRESHOLD': 'd',
        'JUMP_TIMING': 'h',    # Sync | Async
        'INTERLEAVE': 'h',    # On |  This setting is stronger than .
        'ZEROING': 'h',    # On | Off
        'COUPLING': 'h',    # The Off | Pair | All setting is weaker than .
        'RUN_MODE': 'h',    # Continuous | Triggered | Gated | Sequence
        'WAIT_VALUE': 'h',    # First | Last
        'RUN_STATE': 'h',    # On | Off
        'INTERLEAVE_ADJ_PHASE': 'd',
        'INTERLEAVE_ADJ_AMPLITUDE': 'd',
    }
    AWG_FILE_FORMAT_CHANNEL: ClassVar[dict[str, str]] = {
        # Include NULL.(Output Waveform Name for Non-Sequence mode)
        'OUTPUT_WAVEFORM_NAME_N': 's',
        'CHANNEL_STATE_N': 'h',  # On | Off
        'ANALOG_DIRECT_OUTPUT_N': 'h',  # On | Off
        'ANALOG_FILTER_N': 'h',  # Enum type.
        'ANALOG_METHOD_N': 'h',  # Amplitude/Offset, High/Low
        # When the Input Method is High/Low, it is skipped.
        'ANALOG_AMPLITUDE_N': 'd',
        # When the Input Method is High/Low, it is skipped.
        'ANALOG_OFFSET_N': 'd',
        # When the Input Method is Amplitude/Offset, it is skipped.
        'ANALOG_HIGH_N': 'd',
        # When the Input Method is Amplitude/Offset, it is skipped.
        'ANALOG_LOW_N': 'd',
        'MARKER1_SKEW_N': 'd',
        'MARKER1_METHOD_N': 'h',  # Amplitude/Offset, High/Low
        # When the Input Method is High/Low, it is skipped.
        'MARKER1_AMPLITUDE_N': 'd',
        # When the Input Method is High/Low, it is skipped.
        'MARKER1_OFFSET_N': 'd',
        # When the Input Method is Amplitude/Offset, it is skipped.
        'MARKER1_HIGH_N': 'd',
        # When the Input Method is Amplitude/Offset, it is skipped.
        'MARKER1_LOW_N': 'd',
        'MARKER2_SKEW_N': 'd',
        'MARKER2_METHOD_N': 'h',  # Amplitude/Offset, High/Low
        # When the Input Method is High/Low, it is skipped.
        'MARKER2_AMPLITUDE_N': 'd',
        # When the Input Method is High/Low, it is skipped.
        'MARKER2_OFFSET_N': 'd',
        # When the Input Method is Amplitude/Offset, it is skipped.
        'MARKER2_HIGH_N': 'd',
        # When the Input Method is Amplitude/Offset, it is skipped.
        'MARKER2_LOW_N': 'd',
        'DIGITAL_METHOD_N': 'h',  # Amplitude/Offset, High/Low
        # When the Input Method is High/Low, it is skipped.
        'DIGITAL_AMPLITUDE_N': 'd',
        # When the Input Method is High/Low, it is skipped.
        'DIGITAL_OFFSET_N': 'd',
        # When the Input Method is Amplitude/Offset, it is skipped.
        'DIGITAL_HIGH_N': 'd',
        # When the Input Method is Amplitude/Offset, it is skipped.
        'DIGITAL_LOW_N': 'd',
        'EXTERNAL_ADD_N': 'h',  # AWG5000 only
        'PHASE_DELAY_INPUT_METHOD_N':   'h',  # Phase/DelayInme/DelayInints
        'PHASE_N': 'd',  # When the Input Method is not Phase, it is skipped.
        # When the Input Method is not DelayInTime, it is skipped.
        'DELAY_IN_TIME_N': 'd',
        # When the Input Method is not DelayInPoint, it is skipped.
        'DELAY_IN_POINTS_N': 'd',
        'CHANNEL_SKEW_N': 'd',
        'DC_OUTPUT_LEVEL_N': 'd',  # V
    }

    default_timeout = 180

    def __init__(
        self,
        name: str,
        address: str,
        *,
        num_channels: int = 4,
        **kwargs: "Unpack[VisaInstrumentKWArgs]",
    ):
        """
        Initializes the AWG5014.

        Args:
            name: name of the instrument
            address: GPIB or ethernet address as used by VISA
            num_channels: number of channels on the device
            **kwargs: kwargs are forwarded to base class.
        """
        super().__init__(name, address, **kwargs)

        self._address = address
        self.num_channels = num_channels

        self._values: dict[
            str, dict[str, dict[str, Union[np.ndarray, float, None]]]
        ] = {}
        self._values["files"] = {}

        self.add_function('reset', call_cmd='*RST')

        self.state: Parameter = self.add_parameter("state", get_cmd=self.get_state)
        """Parameter state"""
        self.run_mode: Parameter = self.add_parameter(
            "run_mode",
            get_cmd="AWGControl:RMODe?",
            set_cmd="AWGControl:RMODe " + "{}",
            vals=vals.Enum("CONT", "TRIG", "SEQ", "GAT"),
            get_parser=self.newlinestripper,
        )
        """Parameter run_mode"""
        self.clock_source: Parameter = self.add_parameter(
            "clock_source",
            label="Clock source",
            get_cmd="AWGControl:CLOCk:SOURce?",
            set_cmd="AWGControl:CLOCk:SOURce " + "{}",
            vals=vals.Enum("INT", "EXT"),
            get_parser=self.newlinestripper,
        )
        """Parameter clock_source"""

        self.ref_source: Parameter = self.add_parameter(
            "ref_source",
            label="Reference source",
            get_cmd="SOURce1:ROSCillator:SOURce?",
            set_cmd="SOURce1:ROSCillator:SOURce " + "{}",
            vals=vals.Enum("INT", "EXT"),
            get_parser=self.newlinestripper,
        )
        """Parameter ref_source"""

        self.DC_output: Parameter = self.add_parameter(
            "DC_output",
            label="DC Output (ON/OFF)",
            get_cmd="AWGControl:DC:STATe?",
            set_cmd="AWGControl:DC:STATe {}",
            vals=vals.Ints(0, 1),
            get_parser=int,
        )
        """Parameter DC_output"""

        # sequence parameter(s)
        self.sequence_length: Parameter = self.add_parameter(
            "sequence_length",
            label="Sequence length",
            get_cmd="SEQuence:LENGth?",
            set_cmd="SEQuence:LENGth " + "{}",
            get_parser=int,
            vals=vals.Ints(0, 8000),
            docstring=(
                """
                               This command sets the sequence length.
                               Use this command to create an
                               uninitialized sequence. You can also
                               use the command to clear all sequence
                               elements in a single action by passing
                               0 as the parameter. However, this
                               action cannot be undone so exercise
                               necessary caution. Also note that
                               passing a value less than the
                               sequence’s current length will cause
                               some sequence elements to be deleted at
                               the end of the sequence. For example if
                               self.get_sq_length returns 200 and you
                               subsequently set sequence_length to 21,
                               all sequence elements except the first
                               20 will be deleted.
                               """
            ),
        )
        """
                               This command sets the sequence length.
                               Use this command to create an
                               uninitialized sequence. You can also
                               use the command to clear all sequence
                               elements in a single action by passing
                               0 as the parameter. However, this
                               action cannot be undone so exercise
                               necessary caution. Also note that
                               passing a value less than the
                               sequence’s current length will cause
                               some sequence elements to be deleted at
                               the end of the sequence. For example if
                               self.get_sq_length returns 200 and you
                               subsequently set sequence_length to 21,
                               all sequence elements except the first
                               20 will be deleted.
                               """

        self.sequence_pos: Parameter = self.add_parameter(
            "sequence_pos",
            label="Sequence position",
            get_cmd="AWGControl:SEQuencer:POSition?",
            set_cmd="SEQuence:JUMP:IMMediate {}",
            vals=vals.PermissiveInts(1),
            set_parser=lambda x: int(round(x)),
        )
        """Parameter sequence_pos"""

        # Trigger parameters #
        # Warning: `trigger_mode` is the same as `run_mode`, do not use! exists
        # solely for legacy purposes
        self.trigger_mode: Parameter = self.add_parameter(
            "trigger_mode",
            get_cmd="AWGControl:RMODe?",
            set_cmd="AWGControl:RMODe " + "{}",
            vals=vals.Enum("CONT", "TRIG", "SEQ", "GAT"),
            get_parser=self.newlinestripper,
        )
        """Parameter trigger_mode"""
        self.trigger_impedance: Parameter = self.add_parameter(
            "trigger_impedance",
            label="Trigger impedance",
            unit="Ohm",
            get_cmd="TRIGger:IMPedance?",
            set_cmd="TRIGger:IMPedance " + "{}",
            vals=vals.Enum(50, 1000),
            get_parser=float,
        )
        """Parameter trigger_impedance"""
        self.trigger_level: Parameter = self.add_parameter(
            "trigger_level",
            unit="V",
            label="Trigger level",
            get_cmd="TRIGger:LEVel?",
            set_cmd="TRIGger:LEVel " + "{:.3f}",
            vals=vals.Numbers(-5, 5),
            get_parser=float,
        )
        """Parameter trigger_level"""
        self.trigger_slope: Parameter = self.add_parameter(
            "trigger_slope",
            get_cmd="TRIGger:SLOPe?",
            set_cmd="TRIGger:SLOPe " + "{}",
            vals=vals.Enum("POS", "NEG"),
            get_parser=self.newlinestripper,
        )
        """Parameter trigger_slope"""

        self.trigger_source: Parameter = self.add_parameter(
            "trigger_source",
            get_cmd="TRIGger:SOURce?",
            set_cmd="TRIGger:SOURce " + "{}",
            vals=vals.Enum("INT", "EXT"),
            get_parser=self.newlinestripper,
        )
        """Parameter trigger_source"""

        # Event parameters
        self.event_polarity: Parameter = self.add_parameter(
            "event_polarity",
            get_cmd="EVENt:POL?",
            set_cmd="EVENt:POL " + "{}",
            vals=vals.Enum("POS", "NEG"),
            get_parser=self.newlinestripper,
        )
        """Parameter event_polarity"""
        self.event_impedance: Parameter = self.add_parameter(
            "event_impedance",
            label="Event impedance",
            unit="Ohm",
            get_cmd="EVENt:IMPedance?",
            set_cmd="EVENt:IMPedance " + "{}",
            vals=vals.Enum(50, 1000),
            get_parser=float,
        )
        """Parameter event_impedance"""
        self.event_level: Parameter = self.add_parameter(
            "event_level",
            label="Event level",
            unit="V",
            get_cmd="EVENt:LEVel?",
            set_cmd="EVENt:LEVel " + "{:.3f}",
            vals=vals.Numbers(-5, 5),
            get_parser=float,
        )
        """Parameter event_level"""
        self.event_jump_timing: Parameter = self.add_parameter(
            "event_jump_timing",
            get_cmd="EVENt:JTIMing?",
            set_cmd="EVENt:JTIMing {}",
            vals=vals.Enum("SYNC", "ASYNC"),
            get_parser=self.newlinestripper,
        )
        """Parameter event_jump_timing"""

        self.clock_freq: Parameter = self.add_parameter(
            "clock_freq",
            label="Clock frequency",
            unit="Hz",
            get_cmd="SOURce:FREQuency?",
            set_cmd="SOURce:FREQuency " + "{}",
            vals=vals.Numbers(1e6, 1.2e9),
            get_parser=float,
        )
        """Parameter clock_freq"""

        self.setup_filename: Parameter = self.add_parameter(
            "setup_filename", get_cmd="AWGControl:SNAMe?"
        )
        """Parameter setup_filename"""

        # Channel parameters #
        for i in range(1, self.num_channels+1):
            amp_cmd = f'SOURce{i}:VOLTage:LEVel:IMMediate:AMPLitude'
            offset_cmd = f'SOURce{i}:VOLTage:LEVel:IMMediate:OFFS'
            state_cmd = f'OUTPUT{i}:STATE'
            waveform_cmd = f'SOURce{i}:WAVeform'
            directoutput_cmd = f'AWGControl:DOUTput{i}:STATE'
            filter_cmd = f'OUTPut{i}:FILTer:FREQuency'
            add_input_cmd = f'SOURce{i}:COMBine:FEED'
            dc_out_cmd = f'AWGControl:DC{i}:VOLTage:OFFSet'

            # Set channel first to ensure sensible sorting of pars
            self.add_parameter(f'ch{i}_state',
                               label=f'Status channel {i}',
                               get_cmd=state_cmd + '?',
                               set_cmd=state_cmd + ' {}',
                               vals=vals.Ints(0, 1),
                               get_parser=int)
            self.add_parameter(f'ch{i}_amp',
                               label=f'Amplitude channel {i}',
                               unit='Vpp',
                               get_cmd=amp_cmd + '?',
                               set_cmd=amp_cmd + ' {:.6f}',
                               vals=vals.Numbers(0.02, 4.5),
                               get_parser=float)
            self.add_parameter(f'ch{i}_offset',
                               label=f'Offset channel {i}',
                               unit='V',
                               get_cmd=offset_cmd + '?',
                               set_cmd=offset_cmd + ' {:.3f}',
                               vals=vals.Numbers(-2.25, 2.25),
                               get_parser=float)
            self.add_parameter(f'ch{i}_waveform',
                               label=f'Waveform channel {i}',
                               get_cmd=waveform_cmd + '?',
                               set_cmd=waveform_cmd + ' "{}"',
                               vals=vals.Strings(),
                               get_parser=parsestr)
            self.add_parameter(f'ch{i}_direct_output',
                               label=f'Direct output channel {i}',
                               get_cmd=directoutput_cmd + '?',
                               set_cmd=directoutput_cmd + ' {}',
                               vals=vals.Ints(0, 1))
            self.add_parameter(f'ch{i}_add_input',
                               label='Add input channel {}',
                               get_cmd=add_input_cmd + '?',
                               set_cmd=add_input_cmd + ' {}',
                               vals=vals.Enum('"ESIG"', '"ESIGnal"', '""'),
                               get_parser=self.newlinestripper)
            self.add_parameter(f'ch{i}_filter',
                               label=f'Low pass filter channel {i}',
                               unit='Hz',
                               get_cmd=filter_cmd + '?',
                               set_cmd=filter_cmd + ' {}',
                               vals=vals.Enum(20e6, 100e6,
                                              float('inf'),
                                              'INF', 'INFinity'),
                               get_parser=self._tek_outofrange_get_parser)
            self.add_parameter(f'ch{i}_DC_out',
                               label=f'DC output level channel {i}',
                               unit='V',
                               get_cmd=dc_out_cmd + '?',
                               set_cmd=dc_out_cmd + ' {}',
                               vals=vals.Numbers(-3, 5),
                               get_parser=float)

            # Marker channels
            for j in range(1, 3):
                m_del_cmd = f"SOURce{i}:MARKer{j}:DELay"
                m_high_cmd = f"SOURce{i}:MARKer{j}:VOLTage:LEVel:IMMediate:HIGH"
                m_low_cmd = f"SOURce{i}:MARKer{j}:VOLTage:LEVel:IMMediate:LOW"

                self.add_parameter(
                    f'ch{i}_m{j}_del',
                    label=f'Channel {i} Marker {j} delay',
                    unit='ns',
                    get_cmd=m_del_cmd + '?',
                    set_cmd=m_del_cmd + ' {:.3f}e-9',
                    vals=vals.Numbers(0, 1),
                    get_parser=float)
                self.add_parameter(
                    f'ch{i}_m{j}_high',
                    label=f'Channel {i} Marker {j} high level',
                    unit='V',
                    get_cmd=m_high_cmd + '?',
                    set_cmd=m_high_cmd + ' {:.3f}',
                    vals=vals.Numbers(-0.9, 2.7),
                    get_parser=float)
                self.add_parameter(
                    f'ch{i}_m{j}_low',
                    label=f'Channel {i} Marker {j} low level',
                    unit='V',
                    get_cmd=m_low_cmd + '?',
                    set_cmd=m_low_cmd + ' {:.3f}',
                    vals=vals.Numbers(-1.0, 2.6),
                    get_parser=float)

        self.set('trigger_impedance', 50)
        if self.get('clock_freq') != 1e9:
            log.info('AWG clock freq not set to 1GHz')

        self.connect_message()

    # Convenience parser
    def newlinestripper(self, string: str) -> str:
        if string.endswith('\n'):
            return string[:-1]
        else:
            return string

    def _tek_outofrange_get_parser(self, string: str) -> float:
        val = float(string)
        # note that 9.9e37 is used as a generic out of range value
        # in tektronix instruments
        if val >= 9.9e37:
            val = float('INF')
        return val

    # Functions
    def get_state(self) -> Literal['Idle', 'Waiting for trigger', 'Running']:
        """
        This query returns the run state of the arbitrary waveform
        generator or the sequencer.

        Returns:
            Either 'Idle', 'Waiting for trigger', or 'Running'.

        Raises:
            ValueError: if none of the three states above apply.
        """
        state = self.ask('AWGControl:RSTATe?')
        if state.startswith('0'):
            return 'Idle'
        elif state.startswith('1'):
            return 'Waiting for trigger'
        elif state.startswith('2'):
            return 'Running'
        else:
            raise ValueError(f'{__name__} : AWG in undefined state "{state}"')

    def start(self) -> str:
        """Convenience function, identical to self.run()"""
        return self.run()

    def run(self) -> str:
        """
        This command initiates the output of a waveform or a sequence.
        This is equivalent to pressing Run/Stop button on the front panel.
        The instrument can be put in the run state only when output waveforms
        are assigned to channels.

        Returns:
            The output of self.get_state()
        """
        self.write('AWGControl:RUN')
        return self.get_state()

    def stop(self) -> None:
        """This command stops the output of a waveform or a sequence."""
        self.write('AWGControl:STOP')

    def force_trigger(self) -> None:
        """
        This command generates a trigger event. This is equivalent to
        pressing the Force Trigger button on front panel.
        """
        self.write('*TRG')

    def get_folder_contents(self, print_contents: bool = True) -> str:
        """
        This query returns the current contents and state of the mass storage
        media (on the AWG Windows machine).

        Args:
            print_contents: If True, the folder name and the query
                output are printed. Default: True.

        Returns:
            str: A comma-seperated string of the folder contents.
        """
        contents = self.ask('MMEMory:CATalog?')
        if print_contents:
            print('Current folder:', self.get_current_folder_name())
            print(contents
                  .replace(',"$', '\n$').replace('","', '\n')
                  .replace(',', '\t'))
        return contents

    def get_current_folder_name(self) -> str:
        """
        This query returns the current directory of the file system on the
        arbitrary waveform generator. The current directory for the
        programmatic interface is different from the currently selected
        directory in the Windows Explorer on the instrument.

        Returns:
            A string with the full path of the current folder.
        """
        return self.ask('MMEMory:CDIRectory?')

    def set_current_folder_name(self, file_path: str) -> int:
        """
        Set the current directory of the file system on the arbitrary
        waveform generator. The current directory for the programmatic
        interface is different from the currently selected directory in the
        Windows Explorer on the instrument.

        Args:
            file_path: The full path.

        Returns:
            The number of bytes written to instrument
        """
        writecmd = 'MMEMory:CDIRectory "{}"'
        return self.visa_handle.write(writecmd.format(file_path))

    def change_folder(self, folder: str) -> int:
        """Duplicate of self.set_current_folder_name"""
        writecmd = r'MMEMory:CDIRectory "{}"'
        return self.visa_handle.write(writecmd.format(folder))

    def goto_root(self) -> None:
        """
        Set the current directory of the file system on the arbitrary
        waveform generator to C: (the 'root' location in Windows).
        """
        self.write('MMEMory:CDIRectory "c:\\.."')

    def create_and_goto_dir(self, folder: str) -> str:
        """
        Set the current directory of the file system on the arbitrary
        waveform generator. Creates the directory if if doesn't exist.
        Queries the resulting folder for its contents.

        Args:
            folder: The path of the directory to set as current.
                Note: this function expects only root level directories.

        Returns:
            A comma-seperated string of the folder contents.
        """

        dircheck = f"{folder}, DIR"
        if dircheck in self.get_folder_contents():
            self.change_folder(folder)
            log.debug("Directory already exists")
            log.warning(f"Directory already exists, changed path to {folder}")
            content = self.ask("MMEMory:cat?")
            log.info("Contents of folder is %s", content)
        elif self.get_current_folder_name() == f'"\\{folder}"':
            log.info(f"Directory already set to {folder}")
        else:
            self.write(f'MMEMory:MDIRectory "{folder}"')
            self.write(f'MMEMory:CDIRectory "{folder}"')

        return self.get_folder_contents()

    def all_channels_on(self) -> None:
        """
        Set the state of all channels to be ON. Note: only channels with
        defined waveforms can be ON.
        """
        for i in range(1, self.num_channels+1):
            self.set(f'ch{i}_state', 1)

    def all_channels_off(self) -> None:
        """Set the state of all channels to be OFF."""
        for i in range(1, self.num_channels+1):
            self.set(f'ch{i}_state', 0)

    #####################
    # Sequences section #
    #####################

    def force_trigger_event(self) -> None:
        """
        This command generates a trigger event. Equivalent to
        self.force_trigger.
        """
        self.write('TRIGger:IMMediate')

    def force_event(self) -> None:
        """
        This command generates a forced event. This is used to generate the
        event when the sequence is waiting for an event jump. This is
        equivalent to pressing the Force Event button on the front panel of the
        instrument.
        """
        self.write('EVENt:IMMediate')

    def set_sqel_event_target_index(self, element_no: int, index: int) -> None:
        """
        This command sets the target index for
        the sequencer’s event jump operation. Note that this will take
        effect only when the event jump target type is set to
        INDEX.

        Args:
            element_no: The sequence element number
            index: The index to set the target to
        """
        self.write(f"SEQuence:ELEMent{element_no}:JTARGet:INDex {index}")

    def set_sqel_goto_target_index(
            self,
            element_no: int,
            goto_to_index_no: int
    ) -> None:
        """
        This command sets the target index for the GOTO command of the
        sequencer.  After generating the waveform specified in a
        sequence element, the sequencer jumps to the element specified
        as GOTO target. This is an unconditional jump. If GOTO target
        is not specified, the sequencer simply moves on to the next
        element. If the Loop Count is Infinite, the GOTO target which
        is specified in the element is not used. For this command to
        work, the goto state of the squencer must be ON and the
        sequence element must exist.
        Note that the first element of a sequence is taken to be 1 not 0.


        Args:
            element_no: The sequence element number
            goto_to_index_no: The target index number

        """
        self.write(f"SEQuence:ELEMent{element_no}:GOTO:INDex {goto_to_index_no}")

    def set_sqel_goto_state(self, element_no: int, goto_state: int) -> None:
        """
        This command sets the GOTO state of the sequencer for the specified
        sequence element.

        Args:
            element_no: The sequence element number
            goto_state: The GOTO state of the sequencer. Must be either
                0 (OFF) or 1 (ON).
        """
        allowed_states = [0, 1]
        if goto_state not in allowed_states:
            log.warning(
                f"{goto_state} not recognized as a valid goto"
                " state. Setting to 0 (OFF)."
            )
            goto_state = 0
        self.write(f"SEQuence:ELEMent{element_no}:GOTO:STATe {int(goto_state)}")

    def set_sqel_loopcnt_to_inf(self,
                                element_no: int,
                                state: int = 1) -> None:
        """
        This command sets the infinite looping state for a sequence
        element. When an infinite loop is set on an element, the
        sequencer continuously executes that element. To break the
        infinite loop, issue self.stop()

        Args:
            element_no (int): The sequence element number
            state (int): The infinite loop state. Must be either 0 (OFF) or
                1 (ON).
        """
        allowed_states = [0, 1]
        if state not in allowed_states:
            log.warning(
                f"{state} not recognized as a valid loop state. Setting to 0 (OFF)."
            )
            state = 0

        self.write(f"SEQuence:ELEMent{element_no}:LOOP:INFinite {int(state)}")

    def get_sqel_loopcnt(self, element_no: int = 1) -> str:
        """
        This query returns the loop count (number of repetitions) of a
        sequence element. Loop count setting for an element is ignored
        if the infinite looping state is set to ON.

        Args:
            element_no: The sequence element number. Default: 1.
        """
        return self.ask(f'SEQuence:ELEMent{element_no}:LOOP:COUNt?')

    def set_sqel_loopcnt(self, loopcount: int, element_no: int = 1) -> None:
        """
        This command sets the loop count. Loop count setting for an
        element is ignored if the infinite looping state is set to ON.

        Args:
            loopcount: The number of times the sequence is being output.
                The maximal possible number is 65536, beyond that: infinity.
            element_no: The sequence element number. Default: 1.
        """
        self.write(f"SEQuence:ELEMent{element_no}:LOOP:COUNt {loopcount}")

    def set_sqel_waveform(
            self,
            waveform_name: str,
            channel: int,
            element_no: int = 1
    ) -> None:
        """
        This command sets the waveform for a sequence element on the specified
        channel.

        Args:
            waveform_name: Name of the waveform. Must be in the waveform
                list (either User Defined or Predefined).
            channel: The output channel (1-4)
            element_no: The sequence element number. Default: 1.
        """
        self.write(f'SEQuence:ELEMent{element_no}:WAVeform{channel} "{waveform_name}"')

    def get_sqel_waveform(
            self,
            channel: int,
            element_no: int = 1
    ) -> str:
        """
        This query returns the waveform for a sequence element on the
        specified channel.

        Args:
            channel: The output channel (1-4)
            element_no: The sequence element number. Default: 1.

        Returns:
            The name of the waveform.
        """
        return self.ask(f"SEQuence:ELEMent{element_no}:WAVeform{channel}?")

    def set_sqel_trigger_wait(
            self,
            element_no: int,
            state: int = 1) -> str:
        """
        This command sets the wait trigger state for an element. Send
        a trigger signal in one of the following ways:

          * By using an external trigger signal.
          * By pressing the “Force Trigger” button on the front panel
          * By using self.force_trigger or self.force_trigger_event

        Args:
            element_no: The sequence element number.
            state: The wait trigger state. Must be either 0 (OFF)
                or 1 (ON). Default: 1.

        Returns:
            The current state (after setting it).

        """
        self.write(f'SEQuence:ELEMent{element_no}:TWAit {state}')
        return self.get_sqel_trigger_wait(element_no)

    def get_sqel_trigger_wait(self, element_no: int) -> str:
        """
        This query returns the wait trigger state for an element. Send
        a trigger signal in one of the following ways:

          * By using an external trigger signal.
          * By pressing the “Force Trigger” button on the front panel
          * By using self.force_trigger or self.force_trigger_event

        Args:
            element_no: The sequence element number.

        Returns:
            The current state. Example: '1'.
        """
        return self.ask(f'SEQuence:ELEMent{element_no}:TWAit?')

    def set_sqel_event_jump_target_index(self,
                                         element_no: int,
                                         jtar_index_no: int) -> None:
        """Duplicate of set_sqel_event_target_index"""
        self.write(f"SEQuence:ELEMent{element_no}:JTARget:INDex {jtar_index_no}")

    def set_sqel_event_jump_type(
            self,
            element_no: int,
            jtar_state: str
    ) -> None:
        """
        This command sets the event jump target type for the jump for
        the specified sequence element.  Generate an event in one of
        the following ways:

        * By connecting an external cable to instrument rear panel
          for external event.
        * By pressing the Force Event button on the
          front panel.
        * By using self.force_event

        Args:
            element_no: The sequence element number
            jtar_state: The jump target type. Must be either 'INDEX',
                'NEXT', or 'OFF'.
        """
        self.write(f"SEQuence:ELEMent{element_no}:JTARget:TYPE {jtar_state}")

    def get_sq_mode(self) -> str:
        """
        This query returns the type of the arbitrary waveform
        generator's sequencer. The sequence is executed by the
        hardware sequencer whenever possible.

        Returns:
            str: Either 'HARD' or 'SOFT' indicating that the instrument is in\
              either hardware or software sequencer mode.
        """
        return self.ask('AWGControl:SEQuence:TYPE?')

    ######################
    # AWG file functions #
    ######################

    def _pack_record(
            self,
            name: str,
            value: Union[float, str, Sequence[Any], np.ndarray],
            dtype: str
    ) -> bytes:
        """
        packs awg_file record into a struct in the folowing way:
            struct.pack(fmtstring, namesize, datasize, name, data)
        where fmtstring = '<IIs"dtype"'

        The file record format is as follows:
        Record Name Size:        (32-bit unsigned integer)
        Record Data Size:        (32-bit unsigned integer)
        Record Name:             (ASCII) (Include NULL.)
        Record Data
        For details see "File and Record Format" in the AWG help

        < denotes little-endian encoding, I and other dtypes are format
        characters denoted in the documentation of the struct package

        Args:
            name: Name of the record (Example: 'MAGIC' or 'SAMPLING_RATE')
            value: The value of that record.
            dtype: String specifying the data type of the record.
                Allowed values: 'h', 'd', 's'.
        """
        if len(dtype) == 1:
            record_data = struct.pack("<" + dtype, value)
        elif dtype[-1] == "s":
            assert isinstance(value, str)
            record_data = value.encode("ASCII")
        else:
            assert isinstance(value, (abc.Sequence, np.ndarray))
            if dtype[-1] == "H" and isinstance(value, np.ndarray):
                # numpy conversion is fast
                record_data = value.astype("<u2").tobytes()
            else:
                # argument unpacking is slow
                record_data = struct.pack("<" + dtype, *value)

        # the zero byte at the end the record name is the "(Include NULL.)"
        record_name = name.encode('ASCII') + b'\x00'
        record_name_size = len(record_name)
        record_data_size = len(record_data)
        size_struct = struct.pack('<II', record_name_size, record_data_size)
        packed_record = size_struct + record_name + record_data

        return packed_record

    def generate_sequence_cfg(self) -> dict[str, float]:
        """
        This function is used to generate a config file, that is used when
        generating sequence files, from existing settings in the awg.
        Querying the AWG for these settings takes ~0.7 seconds
        """
        log.info('Generating sequence_cfg')

        AWG_sequence_cfg = {
            'SAMPLING_RATE': self.get('clock_freq'),
            'CLOCK_SOURCE': (1 if self.clock_source().startswith('INT')
                             else 2),  # Internal | External
            'REFERENCE_SOURCE': (1 if self.ref_source().startswith('INT')
                                 else 2),  # Internal | External
            'EXTERNAL_REFERENCE_TYPE':   1,  # Fixed | Variable
            'REFERENCE_CLOCK_FREQUENCY_SELECTION': 1,
            # 10 MHz | 20 MHz | 100 MHz
            'TRIGGER_SOURCE':   1 if
            self.get('trigger_source').startswith('EXT') else 2,
            # External | Internal
            "TRIGGER_INPUT_IMPEDANCE": (
                1 if self.get("trigger_impedance") == 50.0 else 2
            ),  # 50 ohm | 1 kohm
            "TRIGGER_INPUT_SLOPE": (
                1 if self.get("trigger_slope").startswith("POS") else 2
            ),  # Positive | Negative
            "TRIGGER_INPUT_POLARITY": (
                1 if self.ask("TRIGger:POLarity?").startswith("POS") else 2
            ),  # Positive | Negative
            "TRIGGER_INPUT_THRESHOLD": self.get("trigger_level"),  # V
            "EVENT_INPUT_IMPEDANCE": (
                1 if self.get("event_impedance") == 50.0 else 2
            ),  # 50 ohm | 1 kohm
            "EVENT_INPUT_POLARITY": (
                1 if self.get("event_polarity").startswith("POS") else 2
            ),  # Positive | Negative
            "EVENT_INPUT_THRESHOLD": self.get("event_level"),  # V
            "JUMP_TIMING": (
                1 if self.get("event_jump_timing").startswith("SYNC") else 2
            ),  # Sync | Async
            "RUN_MODE": 4,  # Continuous | Triggered | Gated | Sequence
            "RUN_STATE": 0,  # On | Off
        }
        return AWG_sequence_cfg

    def generate_channel_cfg(self) -> dict[str, Optional[float]]:
        """
        Function to query if the current channel settings that have
        been changed from their default value and put them in a
        dictionary that can easily be written into an awg file, so as
        to prevent said awg file from falling back to default values.
        (See :meth:`~make_awg_file` and :meth:`~AWG_FILE_FORMAT_CHANNEL`)
        NOTE: This only works for settings changed via the corresponding
        QCoDeS parameter.

        Returns:
            A dict with the current setting for each entry in
            AWG_FILE_FORMAT_HEAD iff this entry applies to the
            AWG5014 AND has been changed from its default value.
        """
        log.info('Getting channel configurations.')

        dirouts = [self.ch1_direct_output.get_latest(),
                   self.ch2_direct_output.get_latest(),
                   self.ch3_direct_output.get_latest(),
                   self.ch4_direct_output.get_latest()]

        # the return value of the parameter is different from what goes
        # into the .awg file, so we translate it
        filtertrans = {20e6: 1, 100e6: 3, 9.9e37: 10,
                       'INF': 10, 'INFinity': 10,
                       float('inf'): 10, None: None}
        filters = [filtertrans[self.ch1_filter.get_latest()],
                   filtertrans[self.ch2_filter.get_latest()],
                   filtertrans[self.ch3_filter.get_latest()],
                   filtertrans[self.ch4_filter.get_latest()]]

        amps = [self.ch1_amp.get_latest(),
                self.ch2_amp.get_latest(),
                self.ch3_amp.get_latest(),
                self.ch4_amp.get_latest()]

        offsets = [self.ch1_offset.get_latest(),
                   self.ch2_offset.get_latest(),
                   self.ch3_offset.get_latest(),
                   self.ch4_offset.get_latest()]

        mrk1highs = [self.ch1_m1_high.get_latest(),
                     self.ch2_m1_high.get_latest(),
                     self.ch3_m1_high.get_latest(),
                     self.ch4_m1_high.get_latest()]

        mrk1lows = [self.ch1_m1_low.get_latest(),
                    self.ch2_m1_low.get_latest(),
                    self.ch3_m1_low.get_latest(),
                    self.ch4_m1_low.get_latest()]

        mrk2highs = [self.ch1_m2_high.get_latest(),
                     self.ch2_m2_high.get_latest(),
                     self.ch3_m2_high.get_latest(),
                     self.ch4_m2_high.get_latest()]

        mrk2lows = [self.ch1_m2_low.get_latest(),
                    self.ch2_m2_low.get_latest(),
                    self.ch3_m2_low.get_latest(),
                    self.ch4_m2_low.get_latest()]

        # the return value of the parameter is different from what goes
        # into the .awg file, so we translate it
        addinptrans = {'"ESIG"': 1, '""': 0, None: None}
        addinputs = [addinptrans[self.ch1_add_input.get_latest()],
                     addinptrans[self.ch2_add_input.get_latest()],
                     addinptrans[self.ch3_add_input.get_latest()],
                     addinptrans[self.ch4_add_input.get_latest()]]

        # the return value of the parameter is different from what goes
        # into the .awg file, so we translate it
        def mrkdeltrans(x: Optional[float]) -> Optional[float]:
            if x is None:
                return None
            else:
                return x * 1e-9
        mrk1delays = [mrkdeltrans(self.ch1_m1_del.get_latest()),
                      mrkdeltrans(self.ch2_m1_del.get_latest()),
                      mrkdeltrans(self.ch3_m1_del.get_latest()),
                      mrkdeltrans(self.ch4_m1_del.get_latest())]
        mrk2delays = [mrkdeltrans(self.ch1_m2_del.get_latest()),
                      mrkdeltrans(self.ch2_m2_del.get_latest()),
                      mrkdeltrans(self.ch3_m2_del.get_latest()),
                      mrkdeltrans(self.ch4_m2_del.get_latest())]

        AWG_channel_cfg: dict[str, Optional[float]] = {}

        for chan in range(1, self.num_channels+1):
            if dirouts[chan - 1] is not None:
                AWG_channel_cfg.update({f'ANALOG_DIRECT_OUTPUT_{chan}':
                                        int(dirouts[chan - 1])})
            if filters[chan - 1] is not None:
                AWG_channel_cfg.update({f'ANALOG_FILTER_{chan}':
                                        filters[chan - 1]})
            if amps[chan - 1] is not None:
                AWG_channel_cfg.update({f'ANALOG_AMPLITUDE_{chan}':
                                        amps[chan - 1]})
            if offsets[chan - 1] is not None:
                AWG_channel_cfg.update({f'ANALOG_OFFSET_{chan}':
                                        offsets[chan - 1]})
            if mrk1highs[chan - 1] is not None:
                AWG_channel_cfg.update({f'MARKER1_HIGH_{chan}':
                                        mrk1highs[chan - 1]})
            if mrk1lows[chan - 1] is not None:
                AWG_channel_cfg.update({f'MARKER1_LOW_{chan}':
                                        mrk1lows[chan - 1]})
            if mrk2highs[chan - 1] is not None:
                AWG_channel_cfg.update({f'MARKER2_HIGH_{chan}':
                                        mrk2highs[chan - 1]})
            if mrk2lows[chan - 1] is not None:
                AWG_channel_cfg.update({f'MARKER2_LOW_{chan}':
                                        mrk2lows[chan - 1]})
            if mrk1delays[chan - 1] is not None:
                AWG_channel_cfg.update({f'MARKER1_SKEW_{chan}':
                                        mrk1delays[chan - 1]})
            if mrk2delays[chan - 1] is not None:
                AWG_channel_cfg.update({f'MARKER2_SKEW_{chan}':
                                        mrk2delays[chan - 1]})
            if addinputs[chan - 1] is not None:
                AWG_channel_cfg.update({f'EXTERNAL_ADD_{chan}':
                                        addinputs[chan - 1]})

        return AWG_channel_cfg

    @staticmethod
    def parse_marker_channel_name(name: str) -> _MarkerDescriptor:
        """
        returns from the channel index and marker index from a marker
        descriptor string e.g. '1M1'->(1,1)
        """
        res = re.match(r'^(?P<channel>\d+)M(?P<marker>\d+)$',
                       name)
        assert res is not None

        return _MarkerDescriptor(marker=int(res.group('marker')),
                                 channel=int(res.group('channel')))

    def _generate_awg_file(
        self,
        packed_waveforms: dict[str, np.ndarray],
        wfname_l: np.ndarray,
        nrep: Sequence[int],
        trig_wait: Sequence[int],
        goto_state: Sequence[int],
        jump_to: Sequence[int],
        channel_cfg: dict[str, Any],
        sequence_cfg: Optional[dict[str, float]] = None,
        preservechannelsettings: bool = False,
    ) -> bytes:
        """
        This function generates an .awg-file for uploading to the AWG.
        The .awg-file contains a waveform list, full sequencing information
        and instrument configuration settings.

        Args:
            packed_waveforms: dictionary containing packed waveforms
                with keys wfname_l

            wfname_l: array of waveform names, e.g.
                array([[segm1_ch1,segm2_ch1..], [segm1_ch2,segm2_ch2..],...])

            nrep: list of len(segments) of integers specifying the
                no. of repetions per sequence element.
                Allowed values: 1 to 65536.

            trig_wait: list of len(segments) of integers specifying the
                trigger wait state of each sequence element.
                Allowed values: 0 (OFF) or 1 (ON).

            goto_state: list of len(segments) of integers specifying the
                goto state of each sequence element. Allowed values: 0 to 65536
                (0 means next)

            jump_to: list of len(segments) of integers specifying
                the logic jump state for each sequence element. Allowed values:
                0 (OFF) or 1 (ON).

            channel_cfg: dictionary of valid channel configuration
                records. See self.AWG_FILE_FORMAT_CHANNEL for a complete
                overview of valid configuration parameters.

            preservechannelsettings: If True, the current channel
                settings are queried from the instrument and added to
                channel_cfg (does not overwrite). Default: False.

            sequence_cfg: dictionary of valid head configuration records
                     (see self.AWG_FILE_FORMAT_HEAD)
                     When an awg file is uploaded these settings will be set
                     onto the AWG, any parameter not specified will be set to
                     its default value (even overwriting current settings)

        for info on filestructure and valid record names, see AWG Help,
        File and Record Format (Under 'Record Name List' in Help)
        """
        if preservechannelsettings:
            channel_settings = self.generate_channel_cfg()
            for setting in channel_settings:
                if setting not in channel_cfg:
                    channel_cfg.update({setting: channel_settings[setting]})

        timetuple = tuple(np.array(localtime())[[0, 1, 8, 2, 3, 4, 5, 6, 7]])

        # general settings
        head_str = BytesIO()
        bytes_to_write = (self._pack_record('MAGIC', 5000, 'h') +
                          self._pack_record('VERSION', 1, 'h'))
        head_str.write(bytes_to_write)
        # head_str.write(string(bytes_to_write))

        if sequence_cfg is None:
            sequence_cfg = self.generate_sequence_cfg()

        for k in list(sequence_cfg.keys()):
            if k in self.AWG_FILE_FORMAT_HEAD:
                head_str.write(self._pack_record(k, sequence_cfg[k],
                                                 self.AWG_FILE_FORMAT_HEAD[k]))
            else:
                log.warning(f"AWG: {k} not recognized as valid AWG setting")
        # channel settings
        ch_record_str = BytesIO()
        for k in list(channel_cfg.keys()):
            ch_k = k[:-1] + 'N'
            if ch_k in self.AWG_FILE_FORMAT_CHANNEL:
                pack = self._pack_record(k, channel_cfg[k],
                                         self.AWG_FILE_FORMAT_CHANNEL[ch_k])
                ch_record_str.write(pack)

            else:
                log.warning(f"AWG: {k} not recognized as valid AWG channel setting")

        # waveforms
        ii = 21

        wf_record_str = BytesIO()
        wlist = list(packed_waveforms.keys())
        wlist.sort()
        for wf in wlist:
            wfdat = packed_waveforms[wf]
            lenwfdat = len(wfdat)

            wf_record_str.write(
                self._pack_record(f'WAVEFORM_NAME_{ii}', wf + '\x00',
                                  '{}s'.format(len(wf + '\x00'))) +
                self._pack_record(f'WAVEFORM_TYPE_{ii}', 1, 'h') +
                self._pack_record(f'WAVEFORM_LENGTH_{ii}',
                                  lenwfdat, 'l') +
                self._pack_record(f'WAVEFORM_TIMESTAMP_{ii}',
                                  timetuple[:-1], '8H') +
                self._pack_record(f'WAVEFORM_DATA_{ii}', wfdat,
                                  f'{lenwfdat}H'))
            ii += 1

        # sequence
        kk = 1
        seq_record_str = BytesIO()

        for segment in wfname_l.transpose():

            seq_record_str.write(
                self._pack_record(f'SEQUENCE_WAIT_{kk}',
                                  trig_wait[kk - 1], 'h') +
                self._pack_record(f'SEQUENCE_LOOP_{kk}',
                                  int(nrep[kk - 1]), 'l') +
                self._pack_record(f'SEQUENCE_JUMP_{kk}',
                                  jump_to[kk - 1], 'h') +
                self._pack_record(f'SEQUENCE_GOTO_{kk}',
                                  goto_state[kk - 1], 'h'))
            for wfname in segment:
                if wfname is not None:
                    # TODO (WilliamHPNielsen): maybe infer ch automatically
                    # from the data size?
                    ch = wfname[-1]
                    seq_record_str.write(
                        self._pack_record('SEQUENCE_WAVEFORM_NAME_CH_' + ch
                                          + f'_{kk}', wfname + '\x00',
                                          '{}s'.format(len(wfname + '\x00')))
                    )
            kk += 1

        awg_file = (head_str.getvalue() + ch_record_str.getvalue() +
                    wf_record_str.getvalue() + seq_record_str.getvalue())
        return awg_file

    def send_awg_file(
            self,
            filename: str,
            awg_file: bytes,
            verbose: bool = False) -> None:
        """
        Writes an .awg-file onto the disk of the AWG.
        Overwrites existing files.

        Args:
            filename: The name that the file will get on
                the AWG.
            awg_file: A byte sequence containing the awg_file.
                Usually the output of self.make_awg_file.
            verbose: A boolean to allow/suppress printing of messages
                about the status of the filw writing. Default: False.
        """
        if verbose:
            print('Writing to:',
                  self.ask('MMEMory:CDIRectory?').replace('\n', '\\ '),
                  filename)
        # Header indicating the name and size of the file being send
        name_str = f'MMEMory:DATA "{filename}",'.encode('ASCII')
        size_str = ('#' + str(len(str(len(awg_file)))) +
                    str(len(awg_file))).encode('ASCII')
        mes = name_str + size_str + awg_file
        self.visa_handle.write_raw(mes)

    def load_awg_file(self, filename: str) -> None:
        """
        Loads an .awg-file from the disc of the AWG into the AWG memory.
        This may overwrite all instrument settings, the waveform list, and the
        sequence in the sequencer.

        Args:
            filename: The filename of the .awg-file to load.
        """
        s = f'AWGControl:SREStore "{filename}"'
        b = s.encode(encoding="ASCII")
        log.debug(f'Loading awg file using {s}')
        self.visa_handle.write_raw(b)
        # we must update the appropriate parameter(s) for the sequence
        self.sequence_length.set(self.sequence_length.get())

    def make_awg_file(
            self,
            waveforms: Union[Sequence[Sequence[np.ndarray]], Sequence[np.ndarray]],
            m1s: Union[Sequence[Sequence[np.ndarray]], Sequence[np.ndarray]],
            m2s: Union[Sequence[Sequence[np.ndarray]], Sequence[np.ndarray]],
            nreps: Sequence[int],
            trig_waits: Sequence[int],
            goto_states: Sequence[int],
            jump_tos: Sequence[int],
            channels: Optional[Sequence[int]] = None,
            preservechannelsettings: bool = True) -> bytes:
        """
        Args:
            waveforms: A list of the waveforms to be packed. The list
                should be filled like so:
                [[wfm1ch1, wfm2ch1, ...], [wfm1ch2, wfm2ch2], ...]
                Each waveform should be a numpy array with values in the range
                -1 to 1 (inclusive). If you do not wish to send waveforms to
                channels 1 and 2, use the channels parameter.

            m1s: A list of marker 1's. The list should be filled
                like so:
                [[elem1m1ch1, elem2m1ch1, ...], [elem1m1ch2, elem2m1ch2], ...]
                Each marker should be a numpy array containing only 0's and 1's

            m2s: A list of marker 2's. The list should be filled
                like so:
                [[elem1m2ch1, elem2m2ch1, ...], [elem1m2ch2, elem2m2ch2], ...]
                Each marker should be a numpy array containing only 0's and 1's

            nreps: List of integers specifying the no. of
                repetitions per sequence element.  Allowed values: 0 to
                65536. O corresponds to Infinite repetitions.

            trig_waits: List of len(segments) of integers specifying the
                trigger wait state of each sequence element.
                Allowed values: 0 (OFF) or 1 (ON).

            goto_states: List of len(segments) of integers
                specifying the goto state of each sequence
                element. Allowed values: 0 to 65536 (0 means next)

            jump_tos: List of len(segments) of integers specifying
                the logic jump state for each sequence element. Allowed values:
                0 (OFF) or 1 (ON).

            channels (list): List of channels to send the waveforms to.
                Example: [1, 3, 2]

            preservechannelsettings (bool): If True, the current channel
                settings are found from the parameter history and added to
                the .awg file. Else, channel settings are not written in the
                file and will be reset to factory default when the file is
                loaded. Default: True.
            """
        packed_wfs = {}
        waveform_names = []
        if not isinstance(waveforms[0], abc.Sequence):
            waveforms_int: Sequence[Sequence[np.ndarray]] = [cast(Sequence[np.ndarray], waveforms)]
            m1s_int: Sequence[Sequence[np.ndarray]] = [cast(Sequence[np.ndarray], m1s)]
            m2s_int: Sequence[Sequence[np.ndarray]] = [cast(Sequence[np.ndarray], m2s)]
        else:
            waveforms_int = cast(Sequence[Sequence[np.ndarray]], waveforms)
            m1s_int = cast(Sequence[Sequence[np.ndarray]], m1s)
            m2s_int = cast(Sequence[Sequence[np.ndarray]], m2s)

        for ii in range(len(waveforms_int)):
            namelist = []
            for jj in range(len(waveforms_int[ii])):
                if channels is None:
                    thisname = f"wfm{jj + 1:03d}ch{ii + 1}"
                else:
                    thisname = f"wfm{jj + 1:03d}ch{channels[ii]}"
                namelist.append(thisname)

                package = self._pack_waveform(waveforms_int[ii][jj],
                                              m1s_int[ii][jj],
                                              m2s_int[ii][jj])

                packed_wfs[thisname] = package
            waveform_names.append(namelist)

        wavenamearray = np.array(waveform_names, dtype='str')

        channel_cfg: dict[str, Any] = {}

        return self._generate_awg_file(
            packed_wfs, wavenamearray, nreps, trig_waits, goto_states,
            jump_tos, channel_cfg,
            preservechannelsettings=preservechannelsettings)

    def make_send_and_load_awg_file(
            self,
            waveforms: Sequence[Sequence[np.ndarray]],
            m1s: Sequence[Sequence[np.ndarray]],
            m2s: Sequence[Sequence[np.ndarray]],
            nreps: Sequence[int],
            trig_waits: Sequence[int],
            goto_states: Sequence[int],
            jump_tos: Sequence[int],
            channels: Optional[Sequence[int]] = None,
            filename: str = 'customawgfile.awg',
            preservechannelsettings: bool = True
    ) -> None:
        """
        Makes an .awg-file, sends it to the AWG and loads it. The .awg-file
        is uploaded to C:\\\\Users\\\\OEM\\\\Documents. The waveforms appear in
        the user defined waveform list with names wfm001ch1, wfm002ch1, ...

        Args:
            waveforms: A list of the waveforms to upload. The list
                should be filled like so:
                [[wfm1ch1, wfm2ch1, ...], [wfm1ch2, wfm2ch2], ...]
                Each waveform should be a numpy array with values in the range
                -1 to 1 (inclusive). If you do not wish to send waveforms to
                channels 1 and 2, use the channels parameter.

            m1s: A list of marker 1's. The list should be filled
                like so:
                [[elem1m1ch1, elem2m1ch1, ...], [elem1m1ch2, elem2m1ch2], ...]
                Each marker should be a numpy array containing only 0's and 1's

            m2s: A list of marker 2's. The list should be filled
                like so:
                [[elem1m2ch1, elem2m2ch1, ...], [elem1m2ch2, elem2m2ch2], ...]
                Each marker should be a numpy array containing only 0's and 1's

            nreps: List of integers specifying the no. of
                repetions per sequence element.  Allowed values: 0 to
                65536. 0 corresponds to Infinite repetions.

            trig_waits: List of len(segments) of integers specifying the
                trigger wait state of each sequence element.
                Allowed values: 0 (OFF) or 1 (ON).

            goto_states: List of len(segments) of integers
                specifying the goto state of each sequence
                element. Allowed values: 0 to 65536 (0 means next)

            jump_tos: List of len(segments) of integers specifying
                the logic jump state for each sequence element. Allowed values:
                0 (OFF) or 1 (ON).

            channels: List of channels to send the waveforms to.
                Example: [1, 3, 2]

            filename: The name of the .awg-file. Should end with the .awg
                extension. Default: 'customawgfile.awg'

            preservechannelsettings: If True, the current channel
                settings are found from the parameter history and added to
                the .awg file. Else, channel settings are reset to the factory
                default values. Default: True.
        """

        # waveform names and the dictionary of packed waveforms
        awg_file = self.make_awg_file(
            waveforms, m1s, m2s, nreps, trig_waits,
            goto_states, jump_tos, channels=channels,
            preservechannelsettings=preservechannelsettings)

        # by default, an unusable directory is targeted on the AWG
        self.visa_handle.write('MMEMory:CDIRectory "C:\\Users\\OEM\\Documents"')

        self.send_awg_file(filename, awg_file)
        currentdir = self.visa_handle.query('MMEMory:CDIRectory?')
        currentdir = currentdir.replace('"', '')
        currentdir = currentdir.replace('\n', '\\')
        loadfrom = f'{currentdir}{filename}'
        self.load_awg_file(loadfrom)

    def make_and_save_awg_file(self,
                               waveforms: Sequence[Sequence[np.ndarray]],
                               m1s: Sequence[Sequence[np.ndarray]],
                               m2s: Sequence[Sequence[np.ndarray]],
                               nreps: Sequence[int],
                               trig_waits: Sequence[int],
                               goto_states: Sequence[int],
                               jump_tos: Sequence[int],
                               channels: Optional[Sequence[int]] = None,
                               filename: str = 'customawgfile.awg',
                               preservechannelsettings: bool = True) -> None:
        """
        Makes an .awg-file and saves it locally.

        Args:
            waveforms: A list of the waveforms to upload. The list
                should be filled like so:
                [[wfm1ch1, wfm2ch1, ...], [wfm1ch2, wfm2ch2], ...]
                Each waveform should be a numpy array with values in the range
                -1 to 1 (inclusive). If you do not wish to send waveforms to
                channels 1 and 2, use the channels parameter.

            m1s: A list of marker 1's. The list should be filled
                like so:
                [[elem1m1ch1, elem2m1ch1, ...], [elem1m1ch2, elem2m1ch2], ...]
                Each marker should be a numpy array containing only 0's and 1's

            m2s: A list of marker 2's. The list should be filled
                like so:
                [[elem1m2ch1, elem2m2ch1, ...], [elem1m2ch2, elem2m2ch2], ...]
                Each marker should be a numpy array containing only 0's and 1's

            nreps: List of integers specifying the no. of
                repetions per sequence element.  Allowed values: 0 to
                65536. O corresponds to Infinite repetions.

            trig_waits: List of len(segments) of integers specifying the
                trigger wait state of each sequence element.
                Allowed values: 0 (OFF) or 1 (ON).

            goto_states: List of len(segments) of integers
                specifying the goto state of each sequence
                element. Allowed values: 0 to 65536 (0 means next)

            jump_tos: List of len(segments) of integers specifying
                the logic jump state for each sequence element. Allowed values:
                0 (OFF) or 1 (ON).

            channels: List of channels to send the waveforms to.
                Example: [1, 3, 2]

            preservechannelsettings: If True, the current channel
                settings are found from the parameter history and added to
                the .awg file. Else, channel settings are not written in the
                file and will be reset to factory default when the file is
                loaded. Default: True.

            filename: The full path of the .awg-file. Should end with the
                .awg extension. Default: 'customawgfile.awg'
        """
        awg_file = self.make_awg_file(
            waveforms, m1s, m2s, nreps, trig_waits,
            goto_states, jump_tos, channels=channels,
            preservechannelsettings=preservechannelsettings)
        with open(filename, 'wb') as fid:
            fid.write(awg_file)

    def get_error(self) -> str:
        """
        This function retrieves and returns data from the error and
        event queues.

        Returns:
            String containing the error/event number, the error/event
            description.
        """
        return self.ask('SYSTEM:ERRor:NEXT?')

    def _pack_waveform(
            self,
            wf: np.ndarray,
            m1: np.ndarray,
            m2: np.ndarray
    ) -> np.ndarray:
        """
        Converts/packs a waveform and two markers into a 16-bit format
        according to the AWG Integer format specification.
        The waveform occupies 14 bits and the markers one bit each.
        See Table 2-25 in the Programmer's manual for more information

        Since markers can only be in one of two states, the marker input
        arrays should consist only of 0's and 1's.

        Args:
            wf: A numpy array containing the waveform. The
                data type of wf is unimportant.
            m1: A numpy array containing the first marker.
            m2: A numpy array containing the second marker.

        Returns:
            An array of unsigned 16 bit integers.

        Raises:
            Exception: if the lengths of w, m1, and m2 don't match
            TypeError: if the waveform contains values outside (-1, 1)
            TypeError: if the markers contain values that are not 0 or 1
        """

        # Input validation
        if (not((len(wf) == len(m1)) and (len(m1) == len(m2)))):
            raise Exception('error: sizes of the waveforms do not match')
        if np.min(wf) < -1 or np.max(wf) > 1:
            raise TypeError(
                "Waveform values out of bonds. Allowed values: -1 to 1 (inclusive)"
            )
        if not np.all(np.isin(m1, np.array([0, 1]))):
            raise TypeError(
                "Marker 1 contains invalid values. Only 0 and 1 are allowed"
            )
        if not np.all(np.isin(m2, np.array([0, 1]))):
            raise TypeError(
                "Marker 2 contains invalid values. Only 0 and 1 are allowed"
            )

        # Note: we use np.trunc here rather than np.round
        # as it is an order of magnitude faster
        packed_wf = np.trunc(16384 * m1 + 32768 * m2
                             + wf * 8191 + 8191.5).astype(np.uint16)

        if len(np.where(packed_wf == -1)[0]) > 0:
            print(np.where(packed_wf == -1))
        return packed_wf

    ###########################
    # Waveform file functions #
    ###########################

    def _file_dict(
        self, wf: np.ndarray, m1: np.ndarray, m2: np.ndarray, clock: Optional[float]
    ) -> dict[str, Union[np.ndarray, float, None]]:
        """
        Make a file dictionary as used by self.send_waveform_to_list

        Args:
            wf: A numpy array containing the waveform. The
                data type of wf is unimportant.
            m1: A numpy array containing the first marker.
            m2: A numpy array containing the second marker.
            clock: The desired clock frequency

        Returns:
            dict: A dictionary with keys 'w', 'm1', 'm2', 'clock_freq', and
                'numpoints' and corresponding values.
        """

        outdict = {
            'w': wf,
            'm1': m1,
            'm2': m2,
            'clock_freq': clock,
            'numpoints': len(wf)
        }

        return outdict

    def delete_all_waveforms_from_list(self) -> None:
        """
        Delete all user-defined waveforms in the list in a single
        action. Note that there is no “UNDO” action once the waveforms
        are deleted. Use caution before issuing this command.

        If the deleted waveform(s) is (are) currently loaded into
        waveform memory, it (they) is (are) unloaded. If the RUN state
        of the instrument is ON, the state is turned OFF. If the
        channel is on, it will be switched off.
        """
        self.write('WLISt:WAVeform:DELete ALL')

    def get_filenames(self) -> str:
        """Duplicate of self.get_folder_contents"""
        return self.ask('MMEMory:CATalog?')

    def send_DC_pulse(self,
                      DC_channel_number: int,
                      set_level: float,
                      length: float) -> None:
        """
        Sets the DC level on the specified channel, waits a while and then
        resets it to what it was before.

        Note: Make sure that the output DC state is ON.

        Args:
            DC_channel_number (int): The channel number (1-4).
            set_level (float): The voltage level to set to (V).
            length (float): The time to wait before resetting (s).
        """
        DC_channel_number -= 1
        chandcs = [self.ch1_DC_out, self.ch2_DC_out, self.ch3_DC_out,
                   self.ch4_DC_out]

        restore = chandcs[DC_channel_number].get()
        chandcs[DC_channel_number].set(set_level)
        sleep(length)
        chandcs[DC_channel_number].set(restore)

    def is_awg_ready(self) -> bool:
        """
        Assert if the AWG is ready.

        Returns:
            True, irrespective of anything.
        """
        try:
            self.ask('*OPC?')
        # makes the awg read again if there is a timeout
        except Exception as e:
            log.warning(e)
            log.warning('AWG is not ready')
            self.visa_handle.read()
        return True

    def send_waveform_to_list(
            self,
            w: np.ndarray,
            m1: np.ndarray,
            m2: np.ndarray,
            wfmname: str) -> None:
        """
        Send a single complete waveform directly to the "User defined"
        waveform list (prepend it). The data type of the input arrays
        is unimportant, but the marker arrays must contain only 1's
        and 0's.

        Args:
            w: The waveform
            m1: Marker1
            m2: Marker2
            wfmname: waveform name

        Raises:
            Exception: if the lengths of w, m1, and m2 don't match
            TypeError: if the waveform contains values outside (-1, 1)
            TypeError: if the markers contain values that are not 0 or 1
        """
        log.debug(f'Sending waveform {wfmname} to instrument')
        # Check for errors
        dim = len(w)

        # Input validation
        if (not((len(w) == len(m1)) and (len(m1) == len(m2)))):
            raise Exception('error: sizes of the waveforms do not match')
        if min(w) < -1 or max(w) > 1:
            raise TypeError(
                "Waveform values out of bonds. Allowed values: -1 to 1 (inclusive)"
            )
        if (list(m1).count(0) + list(m1).count(1)) != len(m1):
            raise TypeError(
                "Marker 1 contains invalid values. Only 0 and 1 are allowed"
            )
        if (list(m2).count(0) + list(m2).count(1)) != len(m2):
            raise TypeError(
                "Marker 2 contains invalid values. Only 0 and 1 are allowed"
            )

        self._values['files'][wfmname] = self._file_dict(w, m1, m2, None)

        # if we create a waveform with the same name but different size,
        # it will not get over written
        # Delete the possibly existing file (will do nothing if the file
        # doesn't exist
        s = f'WLISt:WAVeform:DEL "{wfmname}"'
        self.write(s)

        # create the waveform
        s = f'WLISt:WAVeform:NEW "{wfmname}",{dim:d},INTEGER'
        self.write(s)
        # Prepare the data block
        number = ((2**13 - 1) + (2**13 - 1) * w + 2**14 *
                  np.array(m1) + 2**15 * np.array(m2))
        number = number.astype('int')
        ws_array = arr.array('H', number)

        ws = ws_array.tobytes()
        s1_str = f'WLISt:WAVeform:DATA "{wfmname}",'
        s1 = s1_str.encode('UTF-8')
        s3 = ws
        s2_str = '#' + str(len(str(len(s3)))) + str(len(s3))
        s2 = s2_str.encode('UTF-8')

        mes = s1 + s2 + s3
        self.visa_handle.write_raw(mes)

    def clear_message_queue(self, verbose: bool = False) -> None:
        """
        Function to clear up (flush) the VISA message queue of the AWG
        instrument. Reads all messages in the queue.

        Args:
            verbose: If True, the read messages are printed.
                Default: False.
        """
        original_timeout = self.visa_handle.timeout
        self.visa_handle.timeout = 1000  # 1 second as VISA counts in ms
        gotexception = False
        while not gotexception:
            try:
                message = self.visa_handle.read()
                if verbose:
                    print(message)
            except VisaIOError:
                gotexception = True
        self.visa_handle.timeout = original_timeout


class Tektronix_AWG5014(TektronixAWG5014):
    """
    Alias with non-conformant name left for backwards compatibility
    """

    pass
