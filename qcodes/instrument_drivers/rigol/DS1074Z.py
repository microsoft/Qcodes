from typing import Union
import numpy as np


from qcodes import VisaInstrument
from qcodes import InstrumentChannel
from qcodes import Instrument
from qcodes import ChannelList
from qcodes import ParameterWithSetpoints, Parameter
from qcodes.utils.validators import Numbers, Enum
from qcodes.utils.validators import Arrays


class TraceNotReady(Exception):
    pass


class ScopeTrace(ParameterWithSetpoints):

    def __init__(self, name, channel, **kwargs):
        super().__init__(name, **kwargs)
        self._trace_ready = False
        self._channel = channel

    def prepare_curvedata(self):
        """
        Prepare oscilloscope to return the trace
        """
        npts = self.root_instrument.waveform_npoints()
        self.shape = (npts,)
        self._trace_ready = True

    def get_raw(self):
        if not self._trace_ready:
            raise TraceNotReady('Prepare the trace by '
                                'calling "prepare_curvedata" '
                                )
        else:
            trace = self._get_full_trace()
            return trace

    def _get_raw_trace(self):
        """
        set the out type from oscilloscope channels to WORD
        """
        self.root_instrument.write(':WAVeform:FORMat WORD')

        """"
        set the channel from where data will be obtained
        """
        self.root_instrument.data_source(f"CHAN{self._channel}")

        """"
        Obtain the trace 
        """
        raw_trace_val = self.root_instrument.visa_handle.query_binary_values(
            'WAV:DATA?',
            datatype='h',
            is_big_endian=False,
            expect_termination=False)
        return np.array(raw_trace_val)

    def _get_full_trace(self):
        y_ori = self.root_instrument.waveform_yorigin()
        y_increm = self.root_instrument.waveform_yincrem()
        y_ref = self.root_instrument.waveform_yref()
        y_raw = self._get_raw_trace()
        full_data = (y_raw * y_increm) - y_ori - y_ref
        return full_data


class RigolDS1074ZChannel(InstrumentChannel):
    """
    Contains methods and attributes specific to the Rigol
    oscilloscope channels.

    The output trace from each channel of the oscilloscope
    can be obtained using 'get_trace' parameter.
    """

    def __init__(self,
                 parent: Union[Instrument, 'InstrumentChannel'],
                 name: str,
                 channel
                 ):
        super().__init__(parent, name)

        self.add_parameter("vertical_scale",
                           get_cmd=":CHANnel{}:SCALe?".format(channel),
                           set_cmd=":CHANnel{}:SCALe {}".format(channel, "{}"),
                           get_parser=float
                           )

        self.add_parameter("get_trace",
                           channel=channel,
                           parameter_class=ScopeTrace,
                           vals=Arrays(shape=(self.parent.waveform_npoints,)),
                           setpoints=(self.parent.time_axis,),
                           raw=True,
                           unit='V'
                           )


class RigolDrivers(VisaInstrument):
    """
    The QCoDeS drivers for Oscilloscope Rigol DS1074Z.

    Args:
        name = name of the instrument
        address = VISA address of the instrument

     Optional arguments:
        timeout = maximum time for oscilloscope to
                  respond to a command
        terminator =

    """

    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, terminator='\n', timeout=5, **kwargs)

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
                           vals=Enum('EDGe',
                                     'PULSe', 'PULS',
                                     'VIDeo', 'VID',
                                     'PATTern', 'PATT'),
                           get_parser=str
                           )

        # trigger mode type - EDGe,PULSe, SLOPe, VIDeo, PATTern, DURATion

        # trigger source
        self.add_parameter('trigger_level',
                           unit='V',
                           get_cmd=self._get_trigger_level,
                           set_cmd=self._set_trigger_level,
                           vals=Numbers()
                           )

        self.add_parameter('data_source',
                           label='Waveform Data source',
                           get_cmd=':WAVeform:SOURce?',
                           set_cmd=':WAVeform:SOURce {}',
                           vals=Enum(*(['CHANnel{}'.format(i) for i in
                                        range(1, 4 + 1)]
                                       + ['CHAN{}'.format(i)
                                          for i in range(1, 4 + 1)])))

        self.add_parameter('time_axis',
                           unit='s',
                           label='time_axis',
                           set_cmd=False,
                           get_cmd=self._get_time_axis,
                           parameter_class=Parameter,
                           vals=Arrays(shape=(self.waveform_npoints,))
                           )

        channels = ChannelList(self,
                               "Channels",
                               RigolDS1074ZChannel,
                               snapshotable=False
                               )

        for channel_number in range(1, 5):
            channel = RigolDS1074ZChannel(self,
                                          "ch{}".format(channel_number),
                                          channel_number
                                          )
            channels.append(channel)

        channels.lock()
        self.add_submodule('channels', channels)

    def _get_time_axis(self):
        xorigin = self.waveform_xorigin()
        xincrem = self.waveform_xincrem()
        npts = self.waveform_npoints()
        xdata = np.linspace(xorigin, npts * xincrem + xorigin, npts)
        return xdata

    def _get_trigger_level(self):
        _trigger_level = self.root_instrument.\
            ask(f":TRIGger:{self.trigger_mode()}:LEVel?")
        return _trigger_level

    def _set_trigger_level(self, value):
        self.root_instrument.\
            write(f":TRIGger:{self.trigger_mode()}:LEVel  {value}")
