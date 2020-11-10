from typing import Any

import numpy as np
from qcodes import (ChannelList, InstrumentChannel, ParameterWithSetpoints,
                    VisaInstrument)
from qcodes.utils.validators import Arrays, Enum, Numbers


class RigolDS1074ZChannel(InstrumentChannel):
    """
    Contains methods and attributes specific to the Rigol
    oscilloscope channels.

    The output trace from each channel of the oscilloscope
    can be obtained using 'trace' parameter.
    """

    def __init__(self,
                 parent: "DS1074Z",
                 name: str,
                 channel: int
                 ):
        super().__init__(parent, name)
        self.channel = channel

        self.add_parameter("vertical_scale",
                           get_cmd=f":CHANnel{channel}:SCALe?",
                           set_cmd=":CHANnel{}:SCALe {}".format(channel, "{}"),
                           get_parser=float
                           )

        self.add_parameter("trace",
                           get_cmd=self._get_full_trace,
                           vals=Arrays(shape=(self.parent.waveform_npoints,)),
                           setpoints=(self.parent.time_axis,),
                           unit='V',
                           parameter_class=ParameterWithSetpoints,
                           snapshot_value=False
                           )

    def _get_full_trace(self) -> np.ndarray:
        y_ori = self.root_instrument.waveform_yorigin()
        y_increm = self.root_instrument.waveform_yincrem()
        y_ref = self.root_instrument.waveform_yref()
        y_raw = self._get_raw_trace()
        y_raw_shifted = y_raw - y_ori - y_ref
        full_data = np.multiply(y_raw_shifted, y_increm)
        return full_data

    def _get_raw_trace(self) -> np.ndarray:
        # set the out type from oscilloscope channels to WORD
        self.root_instrument.write(':WAVeform:FORMat WORD')

        # set the channel from where data will be obtained
        self.root_instrument.data_source(f"ch{self.channel}")

        # Obtain the trace
        raw_trace_val = self.root_instrument.visa_handle.query_binary_values(
            'WAV:DATA?',
            datatype='h',
            is_big_endian=False,
            expect_termination=False)
        return np.array(raw_trace_val)


class DS1074Z(VisaInstrument):
    """
    The QCoDeS drivers for Oscilloscope Rigol DS1074Z.

    Args:
        name: name of the instrument.
        address: VISA address of the instrument.
        timeout: Seconds to allow for responses.
        terminator: terminator for SCPI commands.
    """
    def __init__(
            self,
            name: str,
            address: str,
            terminator: str = '\n',
            timeout: float = 5,
            **kwargs: Any):
        super().__init__(name, address, terminator=terminator, timeout=timeout,
                         **kwargs)

        self.add_parameter('waveform_xorigin',
                           get_cmd='WAVeform:XORigin?',
                           unit='s',
                           get_parser=float
                           )

        self.add_parameter('waveform_xincrem',
                           get_cmd=':WAVeform:XINCrement?',
                           unit='s',
                           get_parser=float
                           )

        self.add_parameter('waveform_npoints',
                           get_cmd='WAV:POIN?',
                           set_cmd='WAV:POIN {}',
                           unit='s',
                           get_parser=int
                           )

        self.add_parameter('waveform_yorigin',
                           get_cmd='WAVeform:YORigin?',
                           unit='V',
                           get_parser=float
                           )

        self.add_parameter('waveform_yincrem',
                           get_cmd=':WAVeform:YINCrement?',
                           unit='V',
                           get_parser=float
                           )

        self.add_parameter('waveform_yref',
                           get_cmd=':WAVeform:YREFerence?',
                           unit='V',
                           get_parser=float
                           )

        self.add_parameter('trigger_mode',
                           get_cmd=':TRIGger:MODE?',
                           set_cmd=':TRIGger:MODE {}',
                           unit='V',
                           vals=Enum('edge',
                                     'pulse',
                                     'video',
                                     'pattern'
                                     ),
                           get_parser=str
                           )

        # trigger source
        self.add_parameter('trigger_level',
                           unit='V',
                           get_cmd=self._get_trigger_level,
                           set_cmd=self._set_trigger_level,
                           vals=Numbers()
                           )

        self.add_parameter('trigger_edge_source',
                           label='Source channel for the edge trigger',
                           get_cmd=':TRIGger:EDGE:SOURce?',
                           set_cmd=':TRIGger:EDGE:SOURce {}',
                           val_mapping={'ch1': 'CHAN1',
                                        'ch2': 'CHAN2',
                                        'ch3': 'CHAN3',
                                        'ch4': 'CHAN4'
                                        }
                           )

        self.add_parameter('trigger_edge_slope',
                           label='Slope of the edge trigger',
                           get_cmd=':TRIGger:EDGE:SLOPe?',
                           set_cmd=':TRIGger:EDGE:SLOPe {}',
                           vals=Enum('positive', 'negative', 'neither')
                           )

        self.add_parameter('data_source',
                           label='Waveform Data source',
                           get_cmd=':WAVeform:SOURce?',
                           set_cmd=':WAVeform:SOURce {}',
                           val_mapping={'ch1': 'CHAN1',
                                        'ch2': 'CHAN2',
                                        'ch3': 'CHAN3',
                                        'ch4': 'CHAN4'
                                        }
                           )

        self.add_parameter('time_axis',
                           unit='s',
                           label='Time',
                           set_cmd=False,
                           get_cmd=self._get_time_axis,
                           vals=Arrays(shape=(self.waveform_npoints,)),
                           snapshot_value=False
                           )

        channels = ChannelList(self,
                               "channels",
                               RigolDS1074ZChannel,
                               snapshotable=False
                               )

        for channel_number in range(1, 5):
            channel = RigolDS1074ZChannel(self,
                                          f"ch{channel_number}",
                                          channel_number
                                          )
            channels.append(channel)

        channels.lock()
        self.add_submodule('channels', channels)

        self.connect_message()

    def _get_time_axis(self) -> np.ndarray:
        xorigin = self.waveform_xorigin()
        xincrem = self.waveform_xincrem()
        npts = self.waveform_npoints()
        xdata = np.linspace(xorigin, npts * xincrem + xorigin, npts)
        return xdata

    def _get_trigger_level(self) -> str:
        trigger_level = self.root_instrument.ask(f":TRIGger:{self.trigger_mode()}:LEVel?")
        return trigger_level

    def _set_trigger_level(self, value: str) -> None:
        self.root_instrument.write(f":TRIGger:{self.trigger_mode()}:LEVel {value}")
