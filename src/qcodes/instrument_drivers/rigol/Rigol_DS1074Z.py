from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from qcodes.instrument import (
    ChannelList,
    InstrumentBaseKWArgs,
    InstrumentChannel,
    VisaInstrument,
    VisaInstrumentKWArgs,
)
from qcodes.parameters import ParameterWithSetpoints
from qcodes.validators import Arrays, Enum, Numbers

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from qcodes.parameters import Parameter


class RigolDS1074ZChannel(InstrumentChannel):
    """
    Contains methods and attributes specific to the Rigol
    oscilloscope channels.

    The output trace from each channel of the oscilloscope
    can be obtained using 'trace' parameter.
    """

    def __init__(
        self,
        parent: "RigolDS1074Z",
        name: str,
        channel: int,
        **kwargs: "Unpack[InstrumentBaseKWArgs]",
    ):
        super().__init__(parent, name, **kwargs)
        self.channel = channel

        self.vertical_scale: Parameter = self.add_parameter(
            "vertical_scale",
            get_cmd=f":CHANnel{channel}:SCALe?",
            set_cmd=":CHANnel{}:SCALe {}".format(channel, "{}"),
            get_parser=float,
        )
        """Parameter vertical_scale"""

        self.trace: ParameterWithSetpoints = self.add_parameter(
            "trace",
            get_cmd=self._get_full_trace,
            vals=Arrays(shape=(self.parent.waveform_npoints,)),
            setpoints=(self.parent.time_axis,),
            unit="V",
            parameter_class=ParameterWithSetpoints,
            snapshot_value=False,
        )
        """Parameter trace"""

    def _get_full_trace(self) -> npt.NDArray:
        y_ori = self.root_instrument.waveform_yorigin()
        y_increm = self.root_instrument.waveform_yincrem()
        y_ref = self.root_instrument.waveform_yref()
        y_raw = self._get_raw_trace()
        y_raw_shifted = y_raw - y_ori - y_ref
        full_data = np.multiply(y_raw_shifted, y_increm)
        return full_data

    def _get_raw_trace(self) -> npt.NDArray:
        # set the out type from oscilloscope channels to WORD
        self.root_instrument.write(":WAVeform:FORMat WORD")

        # set the channel from where data will be obtained
        self.root_instrument.data_source(f"ch{self.channel}")

        # Obtain the trace
        raw_trace_val = self.root_instrument.visa_handle.query_binary_values(
            "WAV:DATA?", datatype="h", is_big_endian=False, expect_termination=False
        )
        return np.array(raw_trace_val)


class RigolDS1074Z(VisaInstrument):
    """
    The QCoDeS drivers for Oscilloscope Rigol DS1074Z.

    Args:
        name: name of the instrument.
        address: VISA address of the instrument.
        timeout: Seconds to allow for responses.
        terminator: terminator for SCPI commands.

    """

    default_terminator = "\n"
    default_timeout = 5

    def __init__(
        self,
        name: str,
        address: str,
        **kwargs: "Unpack[VisaInstrumentKWArgs]",
    ):
        super().__init__(name, address, **kwargs)

        self.waveform_xorigin: Parameter = self.add_parameter(
            "waveform_xorigin", get_cmd="WAVeform:XORigin?", unit="s", get_parser=float
        )
        """Parameter waveform_xorigin"""

        self.waveform_xincrem: Parameter = self.add_parameter(
            "waveform_xincrem",
            get_cmd=":WAVeform:XINCrement?",
            unit="s",
            get_parser=float,
        )
        """Parameter waveform_xincrem"""

        self.waveform_npoints: Parameter = self.add_parameter(
            "waveform_npoints",
            get_cmd="WAV:POIN?",
            set_cmd="WAV:POIN {}",
            unit="s",
            get_parser=int,
        )
        """Parameter waveform_npoints"""

        self.waveform_yorigin: Parameter = self.add_parameter(
            "waveform_yorigin", get_cmd="WAVeform:YORigin?", unit="V", get_parser=float
        )
        """Parameter waveform_yorigin"""

        self.waveform_yincrem: Parameter = self.add_parameter(
            "waveform_yincrem",
            get_cmd=":WAVeform:YINCrement?",
            unit="V",
            get_parser=float,
        )
        """Parameter waveform_yincrem"""

        self.waveform_yref: Parameter = self.add_parameter(
            "waveform_yref", get_cmd=":WAVeform:YREFerence?", unit="V", get_parser=float
        )
        """Parameter waveform_yref"""

        self.trigger_mode: Parameter = self.add_parameter(
            "trigger_mode",
            get_cmd=":TRIGger:MODE?",
            set_cmd=":TRIGger:MODE {}",
            unit="V",
            vals=Enum("edge", "pulse", "video", "pattern"),
            get_parser=str,
        )
        """Parameter trigger_mode"""

        # trigger source
        self.trigger_level: Parameter = self.add_parameter(
            "trigger_level",
            unit="V",
            get_cmd=self._get_trigger_level,
            set_cmd=self._set_trigger_level,
            vals=Numbers(),
        )
        """Parameter trigger_level"""

        self.trigger_edge_source: Parameter = self.add_parameter(
            "trigger_edge_source",
            label="Source channel for the edge trigger",
            get_cmd=":TRIGger:EDGE:SOURce?",
            set_cmd=":TRIGger:EDGE:SOURce {}",
            val_mapping={
                "ch1": "CHAN1",
                "ch2": "CHAN2",
                "ch3": "CHAN3",
                "ch4": "CHAN4",
            },
        )
        """Parameter trigger_edge_source"""

        self.trigger_edge_slope: Parameter = self.add_parameter(
            "trigger_edge_slope",
            label="Slope of the edge trigger",
            get_cmd=":TRIGger:EDGE:SLOPe?",
            set_cmd=":TRIGger:EDGE:SLOPe {}",
            vals=Enum("positive", "negative", "neither"),
        )
        """Parameter trigger_edge_slope"""

        self.data_source: Parameter = self.add_parameter(
            "data_source",
            label="Waveform Data source",
            get_cmd=":WAVeform:SOURce?",
            set_cmd=":WAVeform:SOURce {}",
            val_mapping={
                "ch1": "CHAN1",
                "ch2": "CHAN2",
                "ch3": "CHAN3",
                "ch4": "CHAN4",
            },
        )
        """Parameter data_source"""

        self.time_axis: Parameter = self.add_parameter(
            "time_axis",
            unit="s",
            label="Time",
            set_cmd=False,
            get_cmd=self._get_time_axis,
            vals=Arrays(shape=(self.waveform_npoints,)),
            snapshot_value=False,
        )
        """Parameter time_axis"""

        channels = ChannelList(
            self, "channels", RigolDS1074ZChannel, snapshotable=False
        )

        for channel_number in range(1, 5):
            channel = RigolDS1074ZChannel(self, f"ch{channel_number}", channel_number)
            channels.append(channel)

        self.add_submodule("channels", channels.to_channel_tuple())

        self.connect_message()

    def _get_time_axis(self) -> npt.NDArray:
        xorigin = self.waveform_xorigin()
        xincrem = self.waveform_xincrem()
        npts = self.waveform_npoints()
        xdata = np.linspace(xorigin, npts * xincrem + xorigin, npts)
        return xdata

    def _get_trigger_level(self) -> str:
        trigger_level = self.root_instrument.ask(
            f":TRIGger:{self.trigger_mode()}:LEVel?"
        )
        return trigger_level

    def _set_trigger_level(self, value: str) -> None:
        self.root_instrument.write(f":TRIGger:{self.trigger_mode()}:LEVel {value}")
