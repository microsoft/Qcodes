from __future__ import annotations

import datetime as dt
import io
import logging
import struct
import time
import xml.etree.ElementTree as ET
import zipfile as zf
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
from broadbean.sequence import InvalidForgedSequenceError, fs_schema
from typing_extensions import deprecated

from qcodes import validators as vals
from qcodes.instrument import (
    ChannelList,
    Instrument,
    InstrumentBaseKWArgs,
    InstrumentChannel,
    VisaInstrument,
    VisaInstrumentKWArgs,
)
from qcodes.parameters import create_on_off_val_mapping
from qcodes.utils import QCoDeSDeprecationWarning

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from typing_extensions import Unpack

    from qcodes.parameters import Parameter

log = logging.getLogger(__name__)

##################################################
#
# SMALL HELPER FUNCTIONS
#


def _parse_string_response(input_str: str) -> str:
    """
    Remove quotation marks from string and return 'N/A'
    if the input is empty
    """
    output = input_str.replace('"', "")
    output = output if output else "N/A"

    return output


##################################################
#
# MODEL DEPENDENT SETTINGS
#
# TODO: it seems that a lot of settings differ between models
# perhaps these dicts should be merged to one

_fg_path_val_map = {
    "5208": {"DC High BW": "DCHB", "DC High Voltage": "DCHV", "AC Direct": "ACD"},
    "70001A": {"direct": "DIR", "DCamplified": "DCAM", "AC": "AC"},
    "70002A": {"direct": "DIR", "DCamplified": "DCAM", "AC": "AC"},
    "70001B": {"direct": "DIR", "DCamplified": "DCAM", "AC": "AC"},
    "70002B": {"direct": "DIR", "DCamplified": "DCAM", "AC": "AC"},
}

# number of markers per channel
_num_of_markers_map = {"5208": 4, "70001A": 2, "70002A": 2, "70001B": 2, "70002B": 2}

# channel resolution
_chan_resolutions = {
    "5208": [12, 13, 14, 15, 16],
    "70001A": [8, 9, 10],
    "70002A": [8, 9, 10],
    "70001B": [8, 9, 10],
    "70002B": [8, 9, 10],
}

# channel resolution docstrings
_chan_resolution_docstrings = {
    "5208": "12 bit resolution allows for four "
    "markers, 13 bit resolution "
    "allows for three, etc. with 16 bit "
    "allowing for ZERO markers",
    "70001A": "8 bit resolution allows for two "
    "markers, 9 bit resolution "
    "allows for one, and 10 bit "
    "does NOT allow for markers ",
    "70002A": "8 bit resolution allows for two "
    "markers, 9 bit resolution "
    "allows for one, and 10 bit "
    "does NOT allow for markers ",
    "70001B": "8 bit resolution allows for two "
    "markers, 9 bit resolution "
    "allows for one, and 10 bit "
    "does NOT allow for markers ",
    "70002B": "8 bit resolution allows for two "
    "markers, 9 bit resolution "
    "allows for one, and 10 bit "
    "does NOT allow for markers ",
}

# channel amplitudes
_chan_amps = {"70001A": 0.5, "70002A": 0.5, "70001B": 0.5, "70002B": 0.5, "5208": 1.5}

# marker ranges
_marker_high = {
    "70001A": (-1.4, 1.4),
    "70002A": (-1.4, 1.4),
    "70001B": (-1.4, 1.4),
    "70002B": (-1.4, 1.4),
    "5208": (-0.5, 1.75),
}
_marker_low = {
    "70001A": (-1.4, 1.4),
    "70002A": (-1.4, 1.4),
    "70001B": (-1.4, 1.4),
    "70002B": (-1.4, 1.4),
    "5208": (-0.3, 1.55),
}


class SRValidator(vals.Validator[float]):
    """
    Validator to validate the AWG clock sample rate
    """

    def __init__(self, awg: TektronixAWG70000Base) -> None:
        """
        Args:
            awg: The parent instrument instance. We need this since sample
                rate validation depends on many clock settings
        """
        self.awg = awg
        if self.awg.model in ["70001A", "70001B"]:
            self._internal_validator = vals.Numbers(1.49e3, 50e9)
            self._freq_multiplier = 4
        elif self.awg.model in ["70002A", "70002B"]:
            self._internal_validator = vals.Numbers(1.49e3, 25e9)
            self._freq_multiplier = 2
        elif self.awg.model == "5208":
            self._internal_validator = vals.Numbers(1.49e3, 2.5e9)
        # no other models are possible, since the __init__ of
        # the AWG70000A raises an error if anything else is given

    def validate(self, value: float, context: str = "") -> None:
        if "Internal" in self.awg.clock_source():
            self._internal_validator.validate(value)
        else:
            ext_freq = self.awg.clock_external_frequency()
            # TODO: I'm not sure what the minimal allowed sample rate is
            # in this case
            validator = vals.Numbers(1.49e3, self._freq_multiplier * ext_freq)
            validator.validate(value)


class Tektronix70000AWGChannel(InstrumentChannel):
    """
    Class to hold a channel of the AWG.
    """

    def __init__(
        self,
        parent: Instrument,
        name: str,
        channel: int,
        **kwargs: Unpack[InstrumentBaseKWArgs],
    ) -> None:
        """
        Args:
            parent: The Instrument instance to which the channel is
                to be attached.
            name: The name used in the DataSet
            channel: The channel number, either 1 or 2.
            **kwargs: Forwarded to base class.
        """

        super().__init__(parent, name, **kwargs)

        self.channel = channel

        num_channels = self.root_instrument.num_channels
        self.model = self.root_instrument.model

        fg = "function generator"

        if channel not in list(range(1, num_channels + 1)):
            raise ValueError("Illegal channel value.")

        self.state: Parameter = self.add_parameter(
            "state",
            label=f"Channel {channel} state",
            get_cmd=f"OUTPut{channel}:STATe?",
            set_cmd=f"OUTPut{channel}:STATe {{}}",
            vals=vals.Ints(0, 1),
            get_parser=int,
        )
        """Parameter state"""

        ##################################################
        # FGEN PARAMETERS

        # TODO: Setting high and low will change this parameter's value
        self.fgen_amplitude: Parameter = self.add_parameter(
            "fgen_amplitude",
            label=f"Channel {channel} {fg} amplitude",
            get_cmd=f"FGEN:CHANnel{channel}:AMPLitude?",
            set_cmd=f"FGEN:CHANnel{channel}:AMPLitude {{}}",
            unit="V",
            vals=vals.Numbers(0, _chan_amps[self.model]),
            get_parser=float,
        )
        """Parameter fgen_amplitude"""

        self.fgen_offset: Parameter = self.add_parameter(
            "fgen_offset",
            label=f"Channel {channel} {fg} offset",
            get_cmd=f"FGEN:CHANnel{channel}:OFFSet?",
            set_cmd=f"FGEN:CHANnel{channel}:OFFSet {{}}",
            unit="V",
            vals=vals.Numbers(0, 0.250),  # depends on ampl.
            get_parser=float,
        )
        """Parameter fgen_offset"""

        self.fgen_frequency: Parameter = self.add_parameter(
            "fgen_frequency",
            label=f"Channel {channel} {fg} frequency",
            get_cmd=f"FGEN:CHANnel{channel}:FREQuency?",
            set_cmd=partial(self._set_fgfreq, channel),
            unit="Hz",
            get_parser=float,
        )
        """Parameter fgen_frequency"""

        self.fgen_dclevel: Parameter = self.add_parameter(
            "fgen_dclevel",
            label=f"Channel {channel} {fg} DC level",
            get_cmd=f"FGEN:CHANnel{channel}:DCLevel?",
            set_cmd=f"FGEN:CHANnel{channel}:DCLevel {{}}",
            unit="V",
            vals=vals.Numbers(-0.25, 0.25),
            get_parser=float,
        )
        """Parameter fgen_dclevel"""

        self.fgen_signalpath: Parameter = self.add_parameter(
            "fgen_signalpath",
            label=f"Channel {channel} {fg} signal path",
            set_cmd=f"FGEN:CHANnel{channel}:PATH {{}}",
            get_cmd=f"FGEN:CHANnel{channel}:PATH?",
            val_mapping=_fg_path_val_map[self.root_instrument.model],
        )
        """Parameter fgen_signalpath"""

        self.fgen_period: Parameter = self.add_parameter(
            "fgen_period",
            label=f"Channel {channel} {fg} period",
            get_cmd=f"FGEN:CHANnel{channel}:PERiod?",
            unit="s",
            get_parser=float,
        )
        """Parameter fgen_period"""

        self.fgen_phase: Parameter = self.add_parameter(
            "fgen_phase",
            label=f"Channel {channel} {fg} phase",
            get_cmd=f"FGEN:CHANnel{channel}:PHASe?",
            set_cmd=f"FGEN:CHANnel{channel}:PHASe {{}}",
            unit="degrees",
            vals=vals.Numbers(-180, 180),
            get_parser=float,
        )
        """Parameter fgen_phase"""

        self.fgen_symmetry: Parameter = self.add_parameter(
            "fgen_symmetry",
            label=f"Channel {channel} {fg} symmetry",
            set_cmd=f"FGEN:CHANnel{channel}:SYMMetry {{}}",
            get_cmd=f"FGEN:CHANnel{channel}:SYMMetry?",
            unit="%",
            vals=vals.Numbers(0, 100),
            get_parser=float,
        )
        """Parameter fgen_symmetry"""

        self.fgen_type: Parameter = self.add_parameter(
            "fgen_type",
            label=f"Channel {channel} {fg} type",
            set_cmd=f"FGEN:CHANnel{channel}:TYPE {{}}",
            get_cmd=f"FGEN:CHANnel{channel}:TYPE?",
            val_mapping={
                "SINE": "SINE",
                "SQUARE": "SQU",
                "TRIANGLE": "TRI",
                "NOISE": "NOIS",
                "DC": "DC",
                "GAUSSIAN": "GAUSS",
                "EXPONENTIALRISE": "EXPR",
                "EXPONENTIALDECAY": "EXPD",
                "NONE": "NONE",
            },
        )
        """Parameter fgen_type"""

        ##################################################
        # AWG PARAMETERS

        # this command internally uses power in dBm
        # the manual claims that this command only works in AC mode
        # (OUTPut[n]:PATH is AC), but I've tested that it does what
        # one would expect in DIR mode.
        self.awg_amplitude: Parameter = self.add_parameter(
            "awg_amplitude",
            label=f"Channel {channel} AWG peak-to-peak amplitude",
            set_cmd=f"SOURCe{channel}:VOLTage {{}}",
            get_cmd=f"SOURce{channel}:VOLTage?",
            unit="V",
            get_parser=float,
            vals=vals.Numbers(0.250, _chan_amps[self.model]),
        )
        """Parameter awg_amplitude"""

        self.offset: Parameter = self.add_parameter(
            "offset",
            label=f"Channel {channel} Offset for DC Output paths",
            set_cmd=f"SOURce{channel}:VOLTage:LEVel:IMMediate:OFFSet {{}}",
            get_cmd=f"SOURce{channel}:VOLTage:LEVel:IMMediate:OFFSet?",
            unit="V",
            get_parser=float,
            vals=vals.Numbers(-2.0, 2.0),
        )
        """Parameter offset"""

        self.assigned_asset: Parameter = self.add_parameter(
            "assigned_asset",
            label=(f"Waveform/sequence assigned to channel {self.channel}"),
            get_cmd=f"SOURCE{self.channel}:CASSet?",
            get_parser=_parse_string_response,
        )
        """Parameter assigned_asset"""

        # markers
        for mrk in range(1, _num_of_markers_map[self.model] + 1):
            self.add_parameter(
                f"marker{mrk}_high",
                label=f"Channel {channel} marker {mrk} high level",
                set_cmd=partial(self._set_marker, channel, mrk, True),
                get_cmd=f"SOURce{channel}:MARKer{mrk}:VOLTage:HIGH?",
                unit="V",
                vals=vals.Numbers(*_marker_high[self.model]),
                get_parser=float,
            )

            self.add_parameter(
                f"marker{mrk}_low",
                label=f"Channel {channel} marker {mrk} low level",
                set_cmd=partial(self._set_marker, channel, mrk, False),
                get_cmd=f"SOURce{channel}:MARKer{mrk}:VOLTage:LOW?",
                unit="V",
                vals=vals.Numbers(*_marker_low[self.model]),
                get_parser=float,
            )

            self.add_parameter(
                f"marker{mrk}_waitvalue",
                label=f"Channel {channel} marker {mrk} wait state",
                set_cmd=f"OUTPut{channel}:WVALue:MARKer{mrk} {{}}",
                get_cmd=f"OUTPut{channel}:WVALue:MARKer{mrk}?",
                vals=vals.Enum("FIRST", "LOW", "HIGH"),
            )

            self.add_parameter(
                name=f"marker{mrk}_stoppedvalue",
                label=f"Channel {channel} marker {mrk} stopped value",
                set_cmd=f"OUTPut{channel}:SVALue:MARKer{mrk} {{}}",
                get_cmd=f"OUTPut{channel}:SVALue:MARKer{mrk}?",
                vals=vals.Enum("OFF", "LOW"),
            )

        ##################################################
        # MISC.

        self.resolution: Parameter = self.add_parameter(
            "resolution",
            label=f"Channel {channel} bit resolution",
            get_cmd=f"SOURce{channel}:DAC:RESolution?",
            set_cmd=f"SOURce{channel}:DAC:RESolution {{}}",
            vals=vals.Enum(*_chan_resolutions[self.model]),
            get_parser=int,
            docstring=_chan_resolution_docstrings[self.model],
        )
        """Parameter resolution"""

    def _set_marker(
        self, channel: int, marker: int, high: bool, voltage: float
    ) -> None:
        """
        Set the marker high/low value and update the low/high value
        """
        if high:
            this = "HIGH"
            other = "low"
        else:
            this = "LOW"
            other = "high"

        self.write(f"SOURce{channel}:MARKer{marker}:VOLTage:{this} {voltage}")
        self.parameters[f"marker{marker}_{other}"].get()

    def _set_fgfreq(self, channel: int, frequency: float) -> None:
        """
        Set the function generator frequency
        """
        functype = self.fgen_type.get()
        if functype in ["SINE", "SQUARE"]:
            max_freq = 12.5e9
        else:
            max_freq = 6.25e9

        # validate
        if frequency < 1 or frequency > max_freq:
            raise ValueError(
                f"Can not set channel {channel} frequency to {frequency} Hz."
                f" Maximum frequency for function type {functype} is {max_freq} "
                "Hz, minimum is 1 Hz"
            )
        else:
            self.root_instrument.write(f"FGEN:CHANnel{channel}:FREQuency {frequency}")

    def setWaveform(self, name: str) -> None:
        """
        Select a waveform from the waveform list to output on this channel

        Args:
            name: The name of the waveform
        """
        if name not in self.root_instrument.waveformList:
            raise ValueError("No such waveform in the waveform list")

        self.root_instrument.write(f'SOURce{self.channel}:CASSet:WAVeform "{name}"')

    def setSequenceTrack(self, seqname: str, tracknr: int) -> None:
        """
        Assign a track from a sequence to this channel.

        Args:
            seqname: Name of the sequence in the sequence list
            tracknr: Which track to use (1 or 2)
        """

        self.root_instrument.write(
            f'SOURCE{self.channel}:CASSet:SEQuence "{seqname}", {tracknr}'
        )

    def clear_asset(self) -> None:
        """
        Clear assigned assets on this channel
        """

        self.root_instrument.write(f"SOURce{self.channel}:CASSet:CLEAR")


AWGChannel = Tektronix70000AWGChannel
"""
Alias for Tektronix70000AWGChannel for backwards compatibility.
"""


class TektronixAWG70000Base(VisaInstrument):
    """
    Base class for QCoDeS drivers for Tektronix AWG70000 series AWG's.

    The drivers for AWG70001A/AWG70001B and AWG70002A/AWG70002B should be
    subclasses of this general class.
    """

    default_terminator = "\n"
    default_timeout = 10

    def __init__(
        self,
        name: str,
        address: str,
        num_channels: int,
        **kwargs: Unpack[VisaInstrumentKWArgs],
    ) -> None:
        """
        Args:
            name: The name used internally by QCoDeS in the DataSet
            address: The VISA resource name of the instrument
            num_channels: Number of channels on the AWG
            **kwargs: kwargs are forwarded to base class.
        """

        self.num_channels = num_channels

        super().__init__(name, address, **kwargs)

        # The 'model' value begins with 'AWG'
        self.model = self.IDN()["model"][3:]

        if self.model not in ["70001A", "70002A", "70001B", "70002B", "5208"]:
            raise ValueError(
                f"Unknown model type: {self.model}. Are you using "
                f"the right driver for your instrument?"
            )

        self.current_directory: Parameter = self.add_parameter(
            "current_directory",
            label="Current file system directory",
            set_cmd='MMEMory:CDIRectory "{}"',
            get_cmd="MMEMory:CDIRectory?",
            vals=vals.Strings(),
        )
        """Parameter current_directory"""

        self.mode: Parameter = self.add_parameter(
            "mode",
            label="Instrument operation mode",
            set_cmd="INSTrument:MODE {}",
            get_cmd="INSTrument:MODE?",
            vals=vals.Enum("AWG", "FGEN"),
        )
        """Parameter mode"""

        ##################################################
        # Clock parameters

        self.sample_rate: Parameter = self.add_parameter(
            "sample_rate",
            label="Clock sample rate",
            set_cmd="CLOCk:SRATe {}",
            get_cmd="CLOCk:SRATe?",
            unit="Sa/s",
            get_parser=float,
            vals=SRValidator(self),
        )
        """Parameter sample_rate"""

        self.clock_source: Parameter = self.add_parameter(
            "clock_source",
            label="Clock source",
            set_cmd="CLOCk:SOURce {}",
            get_cmd="CLOCk:SOURce?",
            val_mapping={
                "Internal": "INT",
                "Internal, 10 MHZ ref.": "EFIX",
                "Internal, variable ref.": "EVAR",
                "External": "EXT",
            },
        )
        """Parameter clock_source"""

        self.clock_external_frequency: Parameter = self.add_parameter(
            "clock_external_frequency",
            label="External clock frequency",
            set_cmd="CLOCk:ECLock:FREQuency {}",
            get_cmd="CLOCk:ECLock:FREQuency?",
            get_parser=float,
            unit="Hz",
            vals=vals.Numbers(6.25e9, 12.5e9),
        )
        """Parameter clock_external_frequency"""

        self.run_state: Parameter = self.add_parameter(
            "run_state",
            label="Run state",
            get_cmd="AWGControl:RSTATe?",
            val_mapping={"Stopped": "0", "Waiting for trigger": "1", "Running": "2"},
        )
        """Parameter run_state"""

        self.all_output_off: Parameter = self.add_parameter(
            "all_output_off",
            label="All Output Off",
            get_cmd="OUTPut:OFF?",
            set_cmd="OUTPut:OFF {}",
            val_mapping=create_on_off_val_mapping(on_val="1", off_val="0"),
        )
        """Parameter all_output_off"""

        add_channel_list = self.num_channels > 2
        # We deem 2 channels too few for a channel list
        if add_channel_list:
            chanlist = ChannelList(
                self, "Channels", Tektronix70000AWGChannel, snapshotable=False
            )

        for ch_num in range(1, num_channels + 1):
            ch_name = f"ch{ch_num}"
            channel = Tektronix70000AWGChannel(self, ch_name, ch_num)
            self.add_submodule(ch_name, channel)
            if add_channel_list:
                # pyright does not seem to understand
                # that this code can only run iff chanliss is created
                chanlist.append(  # pyright: ignore[reportPossiblyUnboundVariable]
                    channel
                )

        if add_channel_list:
            self.add_submodule(
                "channels",
                chanlist.to_channel_tuple(),  # pyright: ignore[reportPossiblyUnboundVariable]
            )

        # Folder on the AWG where to files are uploaded by default
        self.wfmxFileFolder = "\\Users\\OEM\\Documents"
        self.seqxFileFolder = "\\Users\\OEM\\Documents"

        self.current_directory(self.wfmxFileFolder)

        self.connect_message()

    def force_triggerA(self) -> None:
        """
        Force a trigger A event
        """
        self.write("TRIGger:IMMediate ATRigger")

    def force_triggerB(self) -> None:
        """
        Force a trigger B event
        """
        self.write("TRIGger:IMMediate BTRigger")

    def wait_for_operation_to_complete(self) -> None:
        """
        Waits for the latest issued overlapping command to finish
        """
        self.ask("*OPC?")

    def play(self, wait_for_running: bool = True, timeout: float = 10) -> None:
        """
        Run the AWG/Func. Gen. This command is equivalent to pressing the
        play button on the front panel.

        Args:
            wait_for_running: If True, this command is blocking while the
                instrument is getting ready to play
            timeout: The maximal time to wait for the instrument to play.
                Raises an exception is this time is reached.
        """
        self.write("AWGControl:RUN")
        if wait_for_running:
            start_time = time.perf_counter()
            running = False
            while not running:
                time.sleep(0.1)
                running = self.run_state() in ("Running", "Waiting for trigger")
                waited_for = start_time - time.perf_counter()
                if waited_for > timeout:
                    raise RuntimeError(
                        f"Reached timeout ({timeout} s) "
                        "while waiting for instrument to play."
                        " Perhaps some waveform or sequence is"
                        " corrupt?"
                    )

    def stop(self) -> None:
        """
        Stop the output of the instrument. This command is equivalent to
        pressing the stop button on the front panel.
        """
        self.write("AWGControl:STOP")

    @property
    def sequenceList(self) -> list[str]:
        """
        Return the sequence list as a list of strings
        """
        # There is no SLISt:LIST command, so we do it slightly differently
        N = int(self.ask("SLISt:SIZE?"))
        slist = []
        for n in range(1, N + 1):
            resp = self.ask(f"SLISt:NAME? {n}")
            resp = resp.strip()
            resp = resp.replace('"', "")
            slist.append(resp)

        return slist

    @property
    def waveformList(self) -> list[str]:
        """
        Return the waveform list as a list of strings
        """
        respstr = self.ask("WLISt:LIST?")
        respstr = respstr.strip()
        respstr = respstr.replace('"', "")
        resp = respstr.split(",")

        return resp

    def delete_sequence_from_list(self, seqname: str) -> None:
        """
        Delete the specified sequence from the sequence list

        Args:
            seqname: The name of the sequence (as it appears in the sequence
                list, not the file name) to delete
        """
        self.write(f'SLISt:SEQuence:DELete "{seqname}"')

    def clearSequenceList(self) -> None:
        """
        Clear the sequence list
        """
        self.write("SLISt:SEQuence:DELete ALL")

    def clearWaveformList(self) -> None:
        """
        Clear the waveform list
        """
        self.write("WLISt:WAVeform:DELete ALL")

    @staticmethod
    def makeWFMXFile(data: np.ndarray, amplitude: float) -> bytes:
        """
        Compose a WFMX file

        Args:
            data: A numpy array holding the data. Markers can be included.
            amplitude: The peak-to-peak amplitude (V) assumed to be set on the
                channel that will play this waveform. This information is
                needed as the waveform must be rescaled to (-1, 1) where
                -1 will correspond to the channel's min. voltage and 1 to the
                channel's max. voltage.

        Returns:
            The binary .wfmx file, ready to be sent to the instrument.
        """

        shape = np.shape(data)
        if len(shape) == 1:
            N = shape[0]
            markers_included = False
        elif len(shape) in [2, 3, 4]:
            N = shape[1]
            markers_included = True
        else:
            raise ValueError("Input data has too many dimensions!")

        wfmx_hdr_str = TektronixAWG70000Base._makeWFMXFileHeader(
            num_samples=N, markers_included=markers_included
        )
        wfmx_hdr = bytes(wfmx_hdr_str, "ascii")
        wfmx_data = TektronixAWG70000Base._makeWFMXFileBinaryData(data, amplitude)

        wfmx = wfmx_hdr

        wfmx += wfmx_data

        return wfmx

    def sendSEQXFile(self, seqx: bytes, filename: str, path: str | None = None) -> None:
        """
        Send a binary seqx file to the AWG's memory

        Args:
            seqx: The binary seqx file, preferably the output of
                makeSEQXFile.
            filename: The name of the file on the AWG disk, including the
                extension.
            path: The path to the directory where the file should be saved. If
                omitted, seqxFileFolder will be used.
        """
        if not path:
            path = self.seqxFileFolder

        self._sendBinaryFile(seqx, filename, path)

    def sendWFMXFile(self, wfmx: bytes, filename: str, path: str | None = None) -> None:
        """
        Send a binary wfmx file to the AWG's memory

        Args:
            wfmx: The binary wfmx file, preferably the output of
                makeWFMXFile.
            filename: The name of the file on the AWG disk, including the
                extension.
            path: The path to the directory where the file should be saved. If
                omitted, seqxFileFolder will be used.
        """
        if not path:
            path = self.wfmxFileFolder

        self._sendBinaryFile(wfmx, filename, path)

    def _sendBinaryFile(
        self, binfile: bytes, filename: str, path: str, overwrite: bool = True
    ) -> None:
        """
        Send a binary file to the AWG's mass memory (disk).

        Args:
            binfile: The binary file to send.
            filename: The name of the file on the AWG disk, including the
                extension.
            path: The path to the directory where the file should be saved.
            overwrite: If true, the file on disk gets overwritten
        """

        name_str = f'MMEMory:DATA "{filename}"'.encode("ascii")
        len_file = len(binfile)
        len_str = len(str(len_file))  # No. of digits needed to write length
        size_str = (f",#{len_str}{len_file}").encode("ascii")

        msg = name_str + size_str + binfile

        # IEEE 488.2 limit on a single write is 999,999,999 bytes
        # TODO: If this happens, we should split the file
        if len(msg) > 1e9 - 1:
            raise ValueError("File too large to transfer")

        self.current_directory(path)

        if overwrite:
            self.log.debug(f"Pre-deleting file {filename} at {path}")
            self.visa_handle.write(f'MMEMory:DELete "{filename}"')
            # if the file does not exist,
            # an error code -256 is put in the error queue
            resp = self.visa_handle.query("SYSTem:ERRor:CODE?")
            self.log.debug(f"Pre-deletion finished with return code {resp}")

        self.visa_handle.write_raw(msg)

    def loadWFMXFile(self, filename: str, path: str | None = None) -> None:
        """
        Loads a wfmx from memory into the waveform list
        Only loading from the C: drive is supported

        Args:
            filename: Name of the file (with extension)
            path: Path to load from. If omitted, the default path
                (self.wfmxFileFolder) is used.
        """

        if not path:
            path = self.wfmxFileFolder

        pathstr = "C:" + path + "\\" + filename

        self.write(f'MMEMory:OPEN "{pathstr}"')
        # the above command is overlapping, but we want a blocking command
        self.ask("*OPC?")

    def loadSEQXFile(self, filename: str, path: str | None = None) -> None:
        """
        Load a seqx file from instrument disk memory. All sequences in the file
        are loaded into the sequence list.

        Args:
            filename: The name of the sequence file INCLUDING the extension
            path: Path to load from. If omitted, the default path
                (self.seqxFileFolder) is used.
        """
        if not path:
            path = self.seqxFileFolder

        pathstr = f"C:{path}\\{filename}"

        self.write(f'MMEMory:OPEN:SASSet:SEQuence "{pathstr}"')
        # the above command is overlapping, but we want a blocking command
        self.ask("*OPC?")

    @staticmethod
    def _makeWFMXFileHeader(num_samples: int, markers_included: bool) -> str:
        """
        Compiles a valid XML header for a .wfmx file
        There might be behaviour we can't capture

        We always use 9 digits for the number of header character
        """
        offsetdigits = 9

        if not isinstance(num_samples, int):
            raise ValueError("num_samples must be of type int.")

        if num_samples < 2400:
            raise ValueError("num_samples must be at least 2400.")

        # form the timestamp string
        timezone = time.timezone
        tz_m, _ = divmod(timezone, 60)  # returns (minutes, seconds)
        tz_h, tz_m = divmod(tz_m, 60)
        if np.sign(tz_h) == -1:
            signstr = "-"
            tz_h *= -1
        else:
            signstr = "+"
        timestr = dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        timestr += signstr
        timestr += f"{tz_h:02.0f}:{tz_m:02.0f}"

        hdr = ET.Element(
            "DataFile", attrib={"offset": "0" * offsetdigits, "version": "0.1"}
        )
        dsc = ET.SubElement(hdr, "DataSetsCollection")
        dsc.set("xmlns", "http://www.tektronix.com")
        dsc.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        dsc.set(
            "xsi:schemaLocation",
            (
                r"http://www.tektronix.com file:///"
                r"C:\Program%20Files\Tektronix\AWG70000"
                r"\AWG\Schemas\awgDataSets.xsd"
            ),
        )
        datasets = ET.SubElement(dsc, "DataSets")
        datasets.set("version", "1")
        datasets.set("xmlns", "http://www.tektronix.com")

        # Description of the data
        datadesc = ET.SubElement(datasets, "DataDescription")
        temp_elem = ET.SubElement(datadesc, "NumberSamples")
        temp_elem.text = f"{num_samples:d}"
        temp_elem = ET.SubElement(datadesc, "SamplesType")
        temp_elem.text = "AWGWaveformSample"
        temp_elem = ET.SubElement(datadesc, "MarkersIncluded")
        temp_elem.text = (f"{markers_included}").lower()
        temp_elem = ET.SubElement(datadesc, "NumberFormat")
        temp_elem.text = "Single"
        temp_elem = ET.SubElement(datadesc, "Endian")
        temp_elem.text = "Little"
        temp_elem = ET.SubElement(datadesc, "Timestamp")
        temp_elem.text = timestr

        # Product specific information
        prodspec = ET.SubElement(datasets, "ProductSpecific")
        prodspec.set("name", "")
        temp_elem = ET.SubElement(prodspec, "ReccSamplingRate")
        temp_elem.set("units", "Hz")
        temp_elem.text = "NaN"
        temp_elem = ET.SubElement(prodspec, "ReccAmplitude")
        temp_elem.set("units", "Volts")
        temp_elem.text = "NaN"
        temp_elem = ET.SubElement(prodspec, "ReccOffset")
        temp_elem.set("units", "Volts")
        temp_elem.text = "NaN"
        temp_elem = ET.SubElement(prodspec, "SerialNumber")
        temp_elem = ET.SubElement(prodspec, "SoftwareVersion")
        temp_elem.text = "1.0.0917"
        temp_elem = ET.SubElement(prodspec, "UserNotes")
        temp_elem = ET.SubElement(prodspec, "OriginalBitDepth")
        temp_elem.text = "Floating"
        temp_elem = ET.SubElement(prodspec, "Thumbnail")
        temp_elem = ET.SubElement(prodspec, "CreatorProperties", attrib={"name": ""})
        temp_elem = ET.SubElement(hdr, "Setup")

        xmlstr = ET.tostring(hdr, encoding="unicode")
        xmlstr = xmlstr.replace("><", ">\r\n<")

        # As the final step, count the length of the header and write this
        # in the DataFile tag attribute 'offset'

        xmlstr = xmlstr.replace(
            "0" * offsetdigits,
            "{num:0{pad}d}".format(num=len(xmlstr), pad=offsetdigits),
        )

        return xmlstr

    @staticmethod
    def _makeWFMXFileBinaryData(data: np.ndarray, amplitude: float) -> bytes:
        """
        For the binary part.

        Note that currently only zero markers or two markers are supported;
        one-marker data will break.

        Args:
            data: Either a shape (N,) array with only a waveform or
                a shape (M, N) array with waveform, marker1, marker2, marker3, i.e.
                data = np.array([wfm, m1, ...]). The waveform data is assumed
                to be in V.
            amplitude: The peak-to-peak amplitude (V) assumed to be set on the
                channel that will play this waveform. This information is
                needed as the waveform must be rescaled to (-1, 1) where
                -1 will correspond to the channel's min. voltage and 1 to the
                channel's max. voltage.
        """

        channel_max = amplitude / 2
        channel_min = -amplitude / 2

        shape = np.shape(data)

        if len(shape) == 1:
            N = shape[0]
            binary_marker = b""
            wfm = data
        else:
            N = shape[1]
            M = shape[0]
            wfm = data[0, :]
            markers = data[1, :]
            for i in range(1, M - 1):
                markers += data[i + 1, :] * (2**i)
            markers = markers.astype(int)
            fmt = N * "B"  # endian-ness doesn't matter for one byte
            binary_marker = struct.pack(fmt, *markers)

        if wfm.max() > channel_max or wfm.min() < channel_min:
            log.warning(
                f"Waveform exceeds specified channel range."
                f" The resulting waveform will be clipped. "
                f"Waveform min.: {wfm.min()} (V), waveform max.: {wfm.max()} (V),"
                f"Channel min.: {channel_min} (V), channel max.: {channel_max} (V)"
            )

        # the data must be such that channel_max becomes 1 and
        # channel_min becomes -1
        scale = 2 / amplitude
        wfm = wfm * scale

        # TODO: Is this a fast method?
        fmt = "<" + N * "f"
        binary_wfm = struct.pack(fmt, *wfm)
        binary_out = binary_wfm + binary_marker

        return binary_out

    @staticmethod
    def make_SEQX_from_forged_sequence(
        seq: Mapping[int, Mapping[Any, Any]],
        amplitudes: Sequence[float],
        seqname: str,
        channel_mapping: Mapping[str | int, int] | None = None,
    ) -> bytes:
        """
        Make a .seqx from a forged broadbean sequence.
        Supports subsequences.

        Args:
            seq: The output of broadbean's Sequence.forge()
            amplitudes: A list of the AWG channels' voltage amplitudes.
                The first entry is ch1 etc.
            channel_mapping: A mapping from what the channel is called
                in the broadbean sequence to the integer describing the
                physical channel it should be assigned to.
            seqname: The name that the sequence will have in the AWG's
                sequence list. Used for loading the sequence.

        Returns:
            The binary .seqx file contents. Can be sent directly to the
                instrument or saved on disk.
        """

        try:
            fs_schema.validate(seq)
        except Exception as e:
            raise InvalidForgedSequenceError(e)

        chan_list: list[str | int] = []
        for pos1 in seq.keys():
            for pos2 in seq[pos1]["content"].keys():
                for ch in seq[pos1]["content"][pos2]["data"].keys():
                    if ch not in chan_list:
                        chan_list.append(ch)

        if channel_mapping is None:
            channel_mapping = {ch: ch_ind + 1 for ch_ind, ch in enumerate(chan_list)}

        if len(set(chan_list)) != len(amplitudes):
            raise ValueError("Incorrect number of amplitudes provided.")

        if set(chan_list) != set(channel_mapping.keys()):
            raise ValueError(
                f"Invalid channel_mapping. The sequence has "
                f"channels {set(chan_list)}, but the "
                "channel_mapping maps from the channels "
                f"{set(channel_mapping.keys())}"
            )

        if set(channel_mapping.values()) != set(range(1, 1 + len(chan_list))):
            raise ValueError(
                "Invalid channel_mapping. Must map onto "
                f"{list(range(1, 1+len(chan_list)))}"
            )

        ##########
        # STEP 1:
        # Make all .wfmx files

        wfmx_files: list[bytes] = []
        wfmx_filenames: list[str] = []

        for pos1 in seq.keys():
            for pos2 in seq[pos1]["content"].keys():
                for ch, data in seq[pos1]["content"][pos2]["data"].items():
                    wfm = data["wfm"]

                    markerdata = []
                    for mkey in ["m1", "m2", "m3", "m4"]:
                        if mkey in data.keys():
                            markerdata.append(data.get(mkey))
                    wfm_data = np.stack((wfm, *markerdata))

                    awgchan = channel_mapping[ch]
                    wfmx = TektronixAWG70000Base.makeWFMXFile(
                        wfm_data, amplitudes[awgchan - 1]
                    )
                    wfmx_files.append(wfmx)
                    wfmx_filenames.append(f"wfm_{pos1}_{pos2}_{awgchan}")

        ##########
        # STEP 2:
        # Make all subsequence .sml files

        log.debug(f"Waveforms done: {wfmx_filenames}")

        subseqsml_files: list[str] = []
        subseqsml_filenames: list[str] = []

        for pos1 in seq.keys():
            if seq[pos1]["type"] == "subsequence":
                ss_wfm_names: list[list[str]] = []

                # we need to "flatten" all the individual dicts of element
                # sequence options into one dict of lists of sequencing options
                # and we must also provide default values if nothing
                # is specified
                seqings: list[dict[str, int]] = []
                for pos2 in seq[pos1]["content"].keys():
                    pos_seqs = seq[pos1]["content"][pos2]["sequencing"]
                    pos_seqs["twait"] = pos_seqs.get("twait", 0)
                    pos_seqs["nrep"] = pos_seqs.get("nrep", 1)
                    pos_seqs["jump_input"] = pos_seqs.get("jump_input", 0)
                    pos_seqs["jump_target"] = pos_seqs.get("jump_target", 0)
                    pos_seqs["goto"] = pos_seqs.get("goto", 0)
                    seqings.append(pos_seqs)

                    ss_wfm_names.append(
                        [n for n in wfmx_filenames if f"wfm_{pos1}_{pos2}" in n]
                    )

                seqing = {k: [d[k] for d in seqings] for k in seqings[0].keys()}

                subseqname = f"subsequence_{pos1}"

                log.debug(f"Subsequence waveform names: {ss_wfm_names}")

                subseqsml = TektronixAWG70000Base._makeSMLFile(
                    trig_waits=seqing["twait"],
                    nreps=seqing["nrep"],
                    event_jumps=seqing["jump_input"],
                    event_jump_to=seqing["jump_target"],
                    go_to=seqing["goto"],
                    elem_names=ss_wfm_names,
                    seqname=subseqname,
                    chans=len(channel_mapping),
                )

                subseqsml_files.append(subseqsml)
                subseqsml_filenames.append(f"{subseqname}")

        ##########
        # STEP 3:
        # Make the main .sml file

        asset_names: list[list[str]] = []
        seqings = []
        subseq_positions: list[int] = []
        for pos1 in seq.keys():
            pos_seqs = seq[pos1]["sequencing"]

            pos_seqs["twait"] = pos_seqs.get("twait", 0)
            pos_seqs["nrep"] = pos_seqs.get("nrep", 1)
            pos_seqs["jump_input"] = pos_seqs.get("jump_input", 0)
            pos_seqs["jump_target"] = pos_seqs.get("jump_target", 0)
            pos_seqs["goto"] = pos_seqs.get("goto", 0)
            seqings.append(pos_seqs)
            if seq[pos1]["type"] == "subsequence":
                subseq_positions.append(pos1)
                asset_names.append(
                    [sn for sn in subseqsml_filenames if f"_{pos1}" in sn]
                )
            else:
                asset_names.append([wn for wn in wfmx_filenames if f"wfm_{pos1}" in wn])
        seqing = {k: [d[k] for d in seqings] for k in seqings[0].keys()}

        log.debug(f"Assets for SML file: {asset_names}")

        mainseqname = seqname
        mainseqsml = TektronixAWG70000Base._makeSMLFile(
            trig_waits=seqing["twait"],
            nreps=seqing["nrep"],
            event_jumps=seqing["jump_input"],
            event_jump_to=seqing["jump_target"],
            go_to=seqing["goto"],
            elem_names=asset_names,
            seqname=mainseqname,
            chans=len(channel_mapping),
            subseq_positions=subseq_positions,
        )

        ##########
        # STEP 4:
        # Build the .seqx file

        user_file = b""
        setup_file = TektronixAWG70000Base._makeSetupFile(mainseqname)

        buffer = io.BytesIO()

        zipfile = zf.ZipFile(buffer, mode="a")
        for ssn, ssf in zip(subseqsml_filenames, subseqsml_files):
            zipfile.writestr(f"Sequences/{ssn}.sml", ssf)
        zipfile.writestr(f"Sequences/{mainseqname}.sml", mainseqsml)

        for name, wfile in zip(wfmx_filenames, wfmx_files):
            zipfile.writestr(f"Waveforms/{name}.wfmx", wfile)

        zipfile.writestr("setup.xml", setup_file)
        zipfile.writestr("userNotes.txt", user_file)
        zipfile.close()

        buffer.seek(0)
        seqx = buffer.getvalue()
        buffer.close()

        return seqx

    @staticmethod
    def makeSEQXFile(
        trig_waits: Sequence[int],
        nreps: Sequence[int],
        event_jumps: Sequence[int],
        event_jump_to: Sequence[int],
        go_to: Sequence[int],
        wfms: Sequence[Sequence[np.ndarray]],
        amplitudes: Sequence[float],
        seqname: str,
        flags: Sequence[Sequence[Sequence[int]]] | None = None,
    ) -> bytes:
        """
        Make a full .seqx file (bundle)
        A .seqx file can presumably hold several sequences, but for now
        we support only packing a single sequence

        For a single sequence, a .seqx file is a bundle of two files and
        two folders:

        /Sequences
            sequence.sml

        /Waveforms
            wfm1.wfmx
            wfm2.wfmx
            ...

        setup.xml
        userNotes.txt

        Args:
            trig_waits: Wait for a trigger? If yes, you must specify the
                trigger input. 0 for off, 1 for 'TrigA', 2 for 'TrigB',
                3 for 'Internal'.
            nreps: No. of repetitions. 0 corresponds to infinite.
            event_jumps: Jump when event triggered? If yes, you must specify
                the trigger input. 0 for off, 1 for 'TrigA', 2 for 'TrigB',
                3 for 'Internal'.
            event_jump_to: Jump target in case of event. 1-indexed,
                0 means next. Must be specified for all elements.
            go_to: Which element to play next. 1-indexed, 0 means next.
            wfms: numpy arrays describing each waveform plus two markers,
                packed like np.array([wfm, m1, m2]). These numpy arrays
                are then again packed in lists according to:
                [[wfmch1pos1, wfmch1pos2, ...], [wfmch2pos1, ...], ...]
            amplitudes: The peak-to-peak amplitude in V of the channels, i.e.
                a list [ch1_amp, ch2_amp].
            seqname: The name of the sequence. This name will appear in the
                sequence list. Note that all spaces are converted to '_'
            flags: Flags for the auxiliary outputs. 0 for 'No change', 1 for
                'High', 2 for 'Low', 3 for 'Toggle', or 4 for 'Pulse'. 4 flags
                [A, B, C, D] for every channel in every element, packed like:
                [[ch1pos1, ch1pos2, ...], [ch2pos1, ...], ...]
                If omitted, no flags will be set.

        Returns:
            The binary .seqx file, ready to be sent to the instrument.
        """

        # input sanitising to avoid spaces in filenames
        seqname = seqname.replace(" ", "_")

        (chans, elms) = (len(wfms), len(wfms[0]))
        wfm_names = [
            [f"wfmch{ch}pos{el}" for ch in range(1, chans + 1)]
            for el in range(1, elms + 1)
        ]

        # generate wfmx files for the waveforms
        flat_wfmxs = []
        for amplitude, wfm_lst in zip(amplitudes, wfms):
            flat_wfmxs += [
                TektronixAWG70000Base.makeWFMXFile(wfm, amplitude) for wfm in wfm_lst
            ]

        # This unfortunately assumes no subsequences
        flat_wfm_names = list(
            np.reshape(np.array(wfm_names).transpose(), (chans * elms,))
        )

        sml_file = TektronixAWG70000Base._makeSMLFile(
            trig_waits,
            nreps,
            event_jumps,
            event_jump_to,
            go_to,
            wfm_names,
            seqname,
            chans,
            flags=flags,
        )

        user_file = b""
        setup_file = TektronixAWG70000Base._makeSetupFile(seqname)

        buffer = io.BytesIO()

        zipfile = zf.ZipFile(buffer, mode="a")
        zipfile.writestr(f"Sequences/{seqname}.sml", sml_file)

        for name, wfile in zip(flat_wfm_names, flat_wfmxs):
            zipfile.writestr(f"Waveforms/{name}.wfmx", wfile)

        zipfile.writestr("setup.xml", setup_file)
        zipfile.writestr("userNotes.txt", user_file)
        zipfile.close()

        buffer.seek(0)
        seqx = buffer.getvalue()
        buffer.close()

        return seqx

    @staticmethod
    def _makeSetupFile(sequence: str) -> str:
        """
        Make a setup.xml file.

        Args:
            sequence: The name of the main sequence

        Returns:
            The setup file as a string
        """
        head = ET.Element("RSAPersist")
        head.set("version", "0.1")
        temp_elem = ET.SubElement(head, "Application")
        temp_elem.text = "Pascal"
        temp_elem = ET.SubElement(head, "MainSequence")
        temp_elem.text = sequence
        prodspec = ET.SubElement(head, "ProductSpecific")
        prodspec.set("name", "AWG70002A")
        temp_elem = ET.SubElement(prodspec, "SerialNumber")
        temp_elem.text = "B020397"
        temp_elem = ET.SubElement(prodspec, "SoftwareVersion")
        temp_elem.text = "5.3.0128.0"
        temp_elem = ET.SubElement(prodspec, "CreatorProperties")
        temp_elem.set("name", "")

        xmlstr = ET.tostring(head, encoding="unicode")
        xmlstr = xmlstr.replace("><", ">\r\n<")

        return xmlstr

    @staticmethod
    def _makeSMLFile(
        trig_waits: Sequence[int],
        nreps: Sequence[int],
        event_jumps: Sequence[int],
        event_jump_to: Sequence[int],
        go_to: Sequence[int],
        elem_names: Sequence[Sequence[str]],
        seqname: str,
        chans: int,
        subseq_positions: Sequence[int] = (),
        flags: Sequence[Sequence[Sequence[int]]] | None = None,
    ) -> str:
        """
        Make an xml file describing a sequence.

        Args:
            trig_waits: Wait for a trigger? If yes, you must specify the
                trigger input. 0 for off, 1 for 'TrigA', 2 for 'TrigB',
                3 for 'Internal'.
            nreps: No. of repetitions. 0 corresponds to infinite.
            event_jumps: Jump when event triggered? If yes, you must specify
                the trigger input. 0 for off, 1 for 'TrigA', 2 for 'TrigB',
                3 for 'Internal'.
            event_jump_to: Jump target in case of event. 1-indexed,
                0 means next. Must be specified for all elements.
            go_to: Which element to play next. 1-indexed, 0 means next.
            elem_names: The waveforms/subsequences to use. Should be packed
                like:
                [[wfmpos1ch1, wfmpos1ch2, ...],
                 [subseqpos2],
                 [wfmpos3ch1, wfmpos3ch2, ...], ...]
            seqname: The name of the sequence. This name will appear in
                the sequence list of the instrument.
            chans: The number of channels. Can not be inferred in the case
                of a sequence containing only subsequences, so must be provided
                up front.
            subseq_positions: The positions (step numbers) occupied by
                subsequences
            flags: Flags for the auxiliary outputs. 0 for 'No change', 1 for
                'High', 2 for 'Low', 3 for 'Toggle', or 4 for 'Pulse'. 4 flags
                [A, B, C, D] for every channel in every element, packed like:
                [[ch1pos1, ch1pos2, ...], [ch2pos1, ...], ...]
                If omitted, no flags will be set.

        Returns:
            A str containing the file contents, to be saved as an .sml file
        """

        offsetdigits = 9

        waitinputs = {0: "None", 1: "TrigA", 2: "TrigB", 3: "Internal"}
        eventinputs = {0: "None", 1: "TrigA", 2: "TrigB", 3: "Internal"}
        flaginputs = {0: "NoChange", 1: "High", 2: "Low", 3: "Toggle", 4: "Pulse"}

        inputlsts = [trig_waits, nreps, event_jump_to, go_to]
        lstlens = [len(lst) for lst in inputlsts]
        if lstlens.count(lstlens[0]) != len(lstlens):
            raise ValueError("All input lists must have the same length!")

        if lstlens[0] == 0:
            raise ValueError("Received empty sequence option lengths!")

        if lstlens[0] != len(elem_names):
            raise ValueError(
                "Mismatch between number of waveforms and"
                " number of sequencing steps."
            )

        N = lstlens[0]

        # form the timestamp string
        timezone = time.timezone
        tz_m, _ = divmod(timezone, 60)
        tz_h, tz_m = divmod(tz_m, 60)
        if np.sign(tz_h) == -1:
            signstr = "-"
            tz_h *= -1
        else:
            signstr = "+"
        timestr = dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        timestr += signstr
        timestr += f"{tz_h:02.0f}:{tz_m:02.0f}"

        datafile = ET.Element(
            "DataFile", attrib={"offset": "0" * offsetdigits, "version": "0.1"}
        )
        dsc = ET.SubElement(datafile, "DataSetsCollection")
        dsc.set("xmlns", "http://www.tektronix.com")
        dsc.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        dsc.set(
            "xsi:schemaLocation",
            (
                r"http://www.tektronix.com file:///"
                r"C:\Program%20Files\Tektronix\AWG70000"
                r"\AWG\Schemas\awgSeqDataSets.xsd"
            ),
        )
        datasets = ET.SubElement(dsc, "DataSets")
        datasets.set("version", "1")
        datasets.set("xmlns", "http://www.tektronix.com")

        # Description of the data
        datadesc = ET.SubElement(datasets, "DataDescription")
        temp_elem = ET.SubElement(datadesc, "SequenceName")
        temp_elem.text = seqname
        temp_elem = ET.SubElement(datadesc, "Timestamp")
        temp_elem.text = timestr
        temp_elem = ET.SubElement(datadesc, "JumpTiming")
        temp_elem.text = "JumpImmed"  # TODO: What does this control?
        temp_elem = ET.SubElement(datadesc, "RecSampleRate")
        temp_elem.text = "NaN"
        temp_elem = ET.SubElement(datadesc, "RepeatFlag")
        temp_elem.text = "false"
        temp_elem = ET.SubElement(datadesc, "PatternJumpTable")
        temp_elem.set("Enabled", "false")
        temp_elem.set("Count", "65536")
        steps = ET.SubElement(datadesc, "Steps")
        steps.set("StepCount", f"{N:d}")
        steps.set("TrackCount", f"{chans:d}")

        for n in range(1, N + 1):
            step = ET.SubElement(steps, "Step")
            temp_elem = ET.SubElement(step, "StepNumber")
            temp_elem.text = f"{n:d}"
            # repetitions
            rep = ET.SubElement(step, "Repeat")
            repcount = ET.SubElement(step, "RepeatCount")
            if nreps[n - 1] == 0:
                rep.text = "Infinite"
                repcount.text = "1"
            elif nreps[n - 1] == 1:
                rep.text = "Once"
                repcount.text = "1"
            else:
                rep.text = "RepeatCount"
                repcount.text = f"{nreps[n-1]:d}"
            # trigger wait
            temp_elem = ET.SubElement(step, "WaitInput")
            temp_elem.text = waitinputs[trig_waits[n - 1]]
            # event jump
            temp_elem = ET.SubElement(step, "EventJumpInput")
            temp_elem.text = eventinputs[event_jumps[n - 1]]
            jumpto = ET.SubElement(step, "EventJumpTo")
            jumpstep = ET.SubElement(step, "EventJumpToStep")
            if event_jump_to[n - 1] == 0:
                jumpto.text = "Next"
                jumpstep.text = "1"
            else:
                jumpto.text = "StepIndex"
                jumpstep.text = f"{event_jump_to[n-1]:d}"
            # Go to
            goto = ET.SubElement(step, "GoTo")
            gotostep = ET.SubElement(step, "GoToStep")
            if go_to[n - 1] == 0:
                goto.text = "Next"
                gotostep.text = "1"
            else:
                goto.text = "StepIndex"
                gotostep.text = f"{go_to[n-1]:d}"

            assets = ET.SubElement(step, "Assets")
            for assetname in elem_names[n - 1]:
                asset = ET.SubElement(assets, "Asset")
                temp_elem = ET.SubElement(asset, "AssetName")
                temp_elem.text = assetname
                temp_elem = ET.SubElement(asset, "AssetType")
                if n in subseq_positions:
                    temp_elem.text = "Sequence"
                else:
                    temp_elem.text = "Waveform"

            # convert flag settings to strings
            flags_list = ET.SubElement(step, "Flags")
            for chan in range(chans):
                flagset = ET.SubElement(flags_list, "FlagSet")
                for flgind, flg in enumerate(["A", "B", "C", "D"]):
                    temp_elem = ET.SubElement(flagset, "Flag")
                    temp_elem.set("name", flg)
                    if flags is None:
                        # no flags were passed to the function
                        temp_elem.text = "NoChange"
                    else:
                        temp_elem.text = flaginputs[flags[chan][n - 1][flgind]]

        temp_elem = ET.SubElement(datasets, "ProductSpecific")
        temp_elem.set("name", "")
        temp_elem = ET.SubElement(datafile, "Setup")

        # the tostring() call takes roughly 75% of the total
        # time spent in this function. Can we speed up things?
        # perhaps we should use lxml?
        xmlstr = ET.tostring(datafile, encoding="unicode")
        xmlstr = xmlstr.replace("><", ">\r\n<")

        # As the final step, count the length of the header and write this
        # in the DataFile tag attribute 'offset'

        xmlstr = xmlstr.replace(
            "0" * offsetdigits,
            "{num:0{pad}d}".format(num=len(xmlstr), pad=offsetdigits),
        )

        return xmlstr


@deprecated(
    "Base class renamed TektronixAWG70000Base", category=QCoDeSDeprecationWarning
)
class AWG70000A(TektronixAWG70000Base):
    pass
