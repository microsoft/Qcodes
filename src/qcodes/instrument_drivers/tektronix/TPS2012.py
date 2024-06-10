import binascii
import logging
from functools import partial
from typing import Any

import numpy as np
from pyvisa.errors import VisaIOError
from typing_extensions import TypedDict, Unpack

from qcodes import validators as vals
from qcodes.instrument import (
    ChannelList,
    InstrumentBaseKWArgs,
    InstrumentChannel,
    VisaInstrument,
    VisaInstrumentKWArgs,
)
from qcodes.parameters import ArrayParameter, Parameter, ParamRawDataType

log = logging.getLogger(__name__)


class TraceNotReady(Exception):
    pass


class OutputDict(TypedDict):
    no_of_bytes: int
    no_of_bits: int
    encoding: str
    binary_format: str
    byte_order: str
    no_of_points: int
    waveform_ID: str
    point_format: str
    x_incr: float
    x_zero: float
    x_unit: str
    y_multiplier: float
    y_zero: float
    y_offset: float
    y_unit: str


class ScopeArray(ArrayParameter):
    def __init__(
        self,
        name: str,
        instrument: "TektronixTPS2012Channel",
        channel: int,
        **kwargs: Any,
    ):
        super().__init__(
            name=name,
            shape=(2500,),
            label="Voltage",
            unit="V ",
            setpoint_names=("Time",),
            setpoint_labels=("Time",),
            setpoint_units=("s",),
            docstring="holds an array from scope",
            instrument=instrument,
            **kwargs,
        )
        self.channel = channel

    def calc_set_points(self) -> tuple[np.ndarray, int]:
        assert isinstance(self.instrument, TektronixTPS2012Channel)
        message = self.instrument.ask('WFMPre?')
        preamble = self._preambleparser(message)
        xstart = preamble['x_zero']
        xinc = preamble['x_incr']
        no_of_points = preamble['no_of_points']
        xdata = np.linspace(xstart, no_of_points * xinc + xstart, no_of_points)
        return xdata, no_of_points

    def prepare_curvedata(self) -> None:
        """
        Prepare the scope for returning curve data
        """
        # To calculate set points, we must have the full preamble
        # For the instrument to return the full preamble, the channel
        # in question must be displayed
        assert isinstance(self.instrument, TektronixTPS2012Channel)
        assert isinstance(self.root_instrument, TektronixTPS2012)
        self.instrument.parameters['state'].set('ON')
        self.root_instrument.data_source(f'CH{self.channel}')

        xdata, no_of_points = self.calc_set_points()
        self.setpoints = (tuple(xdata), )
        self.shape = (no_of_points, )

        self.root_instrument.trace_ready = True

    def get_raw(self) -> ParamRawDataType:
        assert isinstance(self.root_instrument, TektronixTPS2012)
        if not self.root_instrument.trace_ready:
            raise TraceNotReady('Please run prepare_curvedata to prepare '
                                'the scope for giving a trace.')
        message = self._curveasker(self.channel)
        _, ydata, _ = self._curveparameterparser(message)
        # Due to the limitations in the current api the below solution
        # to change setpoints does nothing because the setpoints have
        # already been copied to the dataset when get is called.

        # self.setpoints = (tuple(xdata),)
        # self.shape = (npoints,)
        return ydata

    def _curveasker(self, ch: int) -> str:
        assert isinstance(self.instrument, TektronixTPS2012Channel)
        self.instrument.write(f'DATa:SOURce CH{ch}')
        message = self.instrument.ask('WAVFrm?')
        self.instrument.write('*WAI')
        return message

    @staticmethod
    def _binaryparser(curve: str) -> np.ndarray:
        """
        Helper function for parsing the curve data

        Args:
            curve: the return value of 'CURVe?' when
              DATa:ENCdg is set to RPBinary.
              Note: The header and final newline character
              must be removed.

        Returns:
            The curve in units where the digitisation range
            is mapped to (-32768, 32767).
        """
        # TODO: Add support for data width = 1 mode?
        output = np.zeros(int(len(curve)/2))  # data width 2
        # output = np.zeros(int(len(curve)))  # data width 1
        for ii, _ in enumerate(output):
            # casting FTWs
            temp_1 = curve[2*ii:2*ii+1].encode('latin-1')  # data width 2
            temp_2 = binascii.b2a_hex(temp_1)
            temp_3 = (int(temp_2, 16)-128)*256  # data width 2 (1)
            output[ii] = temp_3
        return output

    @staticmethod
    def _preambleparser(response: str) -> OutputDict:
        """
        Parser function for the curve preamble

        Args:
            response: The response of WFMPre?

        Returns:
            A dictionary containing the following keys:
              no_of_bytes, no_of_bits, encoding, binary_format,
              byte_order, no_of_points, waveform_ID, point_format,
              x_incr, x_zero, x_unit, y_multiplier, y_zero, y_offset, y_unit
        """
        response_list = response.split(';')

        outdict: OutputDict = {
            'no_of_bytes': int(response_list[0]),
            'no_of_bits': int(response_list[1]),
            'encoding':  response_list[2],
            'binary_format': response_list[3],
            'byte_order': response_list[4],
            'no_of_points': int(response_list[5]),
            'waveform_ID':  response_list[6],
            'point_format': response_list[7],
            'x_incr': float(response_list[8]),
            # outdict['point_offset'] = response_list[9]  # Always zero
            'x_zero': float(response_list[10]),
            'x_unit': response_list[11],
            'y_multiplier': float(response_list[12]),
            'y_zero': float(response_list[13]),
            'y_offset': float(response_list[14]),
            'y_unit': response_list[15]
        }
        return outdict

    def _curveparameterparser(
        self, waveform: str
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """
        The parser for the curve parameter. Note that WAVFrm? is equivalent
        to WFMPre?; CURVe?

        Args:
            waveform: The return value of WAVFrm?

        Returns:
            Two numpy arrays with the time axis in units
            of s and curve values in units of V; (time, voltages) and
            the number of points as an integer
        """
        fulldata = waveform.split(';')
        preamblestr = ';'.join(fulldata[:16])
        curvestr = ';'.join(fulldata[16:])

        preamble = self._preambleparser(preamblestr)
        # the raw curve data starts with a header containing the char #
        # followed by on digit giving the number of digits in the len of the
        # array in bytes
        # and the length of the array. I.e. the string #45000 is 5000 bytes
        # represented by 4 digits.
        total_number_of_bytes = preamble['no_of_bytes']*preamble['no_of_points']
        raw_data_offset = 2 + len(str(total_number_of_bytes))
        curvestr = curvestr[raw_data_offset:-1]
        rawcurve = self._binaryparser(curvestr)

        yoff = preamble['y_offset']
        yoff -= 2**15  # data width 2
        ymult = preamble['y_multiplier']
        ydata = ymult*(rawcurve-yoff)
        assert len(ydata) == preamble['no_of_points']
        xstart = preamble['x_zero']
        xinc = preamble['x_incr']
        xdata = np.linspace(xstart, len(ydata)*xinc+xstart, len(ydata))
        return xdata, ydata, preamble['no_of_points']


class TektronixTPS2012Channel(InstrumentChannel):
    def __init__(
        self,
        parent: "TektronixTPS2012",
        name: str,
        channel: int,
        **kwargs: Unpack[InstrumentBaseKWArgs],
    ):
        super().__init__(parent, name, **kwargs)

        self.scale: Parameter = self.add_parameter(
            "scale",
            label=f"Channel {channel} Scale",
            unit="V/div",
            get_cmd=f"CH{channel}:SCAle?",
            set_cmd="CH{}:SCAle {}".format(channel, "{}"),
            get_parser=float,
        )
        """Parameter scale"""
        self.position: Parameter = self.add_parameter(
            "position",
            label=f"Channel {channel} Position",
            unit="div",
            get_cmd=f"CH{channel}:POSition?",
            set_cmd="CH{}:POSition {}".format(channel, "{}"),
            get_parser=float,
        )
        """Parameter position"""
        self.curvedata: ScopeArray = self.add_parameter(
            "curvedata",
            channel=channel,
            parameter_class=ScopeArray,
        )
        """Parameter curvedata"""
        self.state: Parameter = self.add_parameter(
            "state",
            label=f"Channel {channel} display state",
            set_cmd="SELect:CH{} {}".format(channel, "{}"),
            get_cmd=partial(self._get_state, channel),
            val_mapping={"ON": 1, "OFF": 0},
            vals=vals.Enum("ON", "OFF"),
        )
        """Parameter state"""

    def _get_state(self, ch: int) -> int:
        """
        get_cmd for the chX_state parameter
        """
        # 'SELect?' returns a ';'-separated string of 0s and 1s
        # denoting state display state of ch1, ch2, ?, ?, ?
        # (maybe ch1, ch2, math, ref1, ref2 ..?)
        selected = list(map(int, self.ask('SELect?').split(';')))
        state = selected[ch - 1]
        return state


TPS2012Channel = TektronixTPS2012Channel


class TektronixTPS2012(VisaInstrument):
    """
    This is the QCoDeS driver for the Tektronix 2012B oscilloscope.
    """

    default_timeout = 20

    def __init__(
        self,
        name: str,
        address: str,
        **kwargs: Unpack[VisaInstrumentKWArgs],
    ):
        """
        Initialises the TPS2012.

        Args:
            name: Name of the instrument used by QCoDeS
            address: Instrument address as used by VISA
            **kwargs: kwargs are forwarded to base class.
        """

        super().__init__(name, address, **kwargs)
        self.connect_message()

        # Scope trace boolean
        self.trace_ready = False

        # functions

        self.add_function('force_trigger',
                          call_cmd='TRIGger FORce',
                          docstring='Force trigger event')
        self.add_function('run',
                          call_cmd='ACQuire:STATE RUN',
                          docstring='Start acquisition')
        self.add_function('stop',
                          call_cmd='ACQuire:STATE STOP',
                          docstring='Stop acquisition')

        # general parameters
        self.trigger_type: Parameter = self.add_parameter(
            "trigger_type",
            label="Type of the trigger",
            get_cmd="TRIGger:MAIn:TYPe?",
            set_cmd="TRIGger:MAIn:TYPe {}",
            vals=vals.Enum("EDGE", "VIDEO", "PULSE"),
        )
        """Parameter trigger_type"""
        self.trigger_source: Parameter = self.add_parameter(
            "trigger_source",
            label="Source for the trigger",
            get_cmd="TRIGger:MAIn:EDGE:SOURce?",
            set_cmd="TRIGger:MAIn:EDGE:SOURce {}",
            vals=vals.Enum("CH1", "CH2"),
        )
        """Parameter trigger_source"""
        self.trigger_edge_slope: Parameter = self.add_parameter(
            "trigger_edge_slope",
            label="Slope for edge trigger",
            get_cmd="TRIGger:MAIn:EDGE:SLOpe?",
            set_cmd="TRIGger:MAIn:EDGE:SLOpe {}",
            vals=vals.Enum("FALL", "RISE"),
        )
        """Parameter trigger_edge_slope"""
        self.trigger_level: Parameter = self.add_parameter(
            "trigger_level",
            label="Trigger level",
            unit="V",
            get_cmd="TRIGger:MAIn:LEVel?",
            set_cmd="TRIGger:MAIn:LEVel {}",
            vals=vals.Numbers(),
        )
        """Parameter trigger_level"""
        self.data_source: Parameter = self.add_parameter(
            "data_source",
            label="Data source",
            get_cmd="DATa:SOUrce?",
            set_cmd="DATa:SOURce {}",
            vals=vals.Enum("CH1", "CH2"),
        )
        """Parameter data_source"""
        self.horizontal_scale: Parameter = self.add_parameter(
            "horizontal_scale",
            label="Horizontal scale",
            unit="s",
            get_cmd="HORizontal:SCAle?",
            set_cmd=self._set_timescale,
            get_parser=float,
            vals=vals.Enum(
                5e-9,
                10e-9,
                25e-9,
                50e-9,
                100e-9,
                250e-9,
                500e-9,
                1e-6,
                2.5e-6,
                5e-6,
                10e-6,
                25e-6,
                50e-6,
                100e-6,
                250e-6,
                500e-6,
                1e-3,
                2.5e-3,
                5e-3,
                10e-3,
                25e-3,
                50e-3,
                100e-3,
                250e-3,
                500e-3,
                1,
                2.5,
                5,
                10,
                25,
                50,
            ),
        )
        """Parameter horizontal_scale"""

        # channel-specific parameters
        channels = ChannelList(
            self, "ScopeChannels", TektronixTPS2012Channel, snapshotable=False
        )
        for ch_num in range(1, 3):
            ch_name = f"ch{ch_num}"
            channel = TektronixTPS2012Channel(self, ch_name, ch_num)
            channels.append(channel)
            self.add_submodule(ch_name, channel)
        self.add_submodule("channels", channels.to_channel_tuple())

        # Necessary settings for parsing the binary curve data
        self.visa_handle.encoding = "latin-1"
        log.info("Set VISA encoding to latin-1")
        self.write("DATa:ENCdg RPBinary")
        log.info("Set TPS2012 data encoding to RPBinary (Positive Integer Binary)")
        self.write("DATa:WIDTh 2")
        log.info("Set TPS2012 data width to 2")
        # Note: using data width 2 has been tested to not add
        # significantly to transfer times. The maximal length
        # of an array in one transfer is 2500 points.

    def _set_timescale(self, scale: float) -> None:
        """
        set_cmd for the horizontal_scale
        """
        self.trace_ready = False
        self.write(f'HORizontal:SCAle {scale}')

    ##################################################
    # METHODS FOR THE USER                           #
    ##################################################

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


class TPS2012(TektronixTPS2012):
    """
    Deprecated alias for ``TektronixTPS2012``
    """

    pass
