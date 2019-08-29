import logging
from typing import Dict, Callable
from functools import partial

import numpy as np

from qcodes import VisaInstrument, validators as vals
from qcodes import InstrumentChannel, ChannelList
from qcodes import ArrayParameter
from qcodes.utils.validators import Enum, Numbers

"""
This is a copy of QCoDeS Infiniium driver adjusted for the 
Keysight Infiniium 54832B oscilloscope for trace measurements.
Modified by R.Savytskyy (29.08.2019)
"""

log = logging.getLogger(__name__)


class Infiniium(VisaInstrument):

    def __init__(self, name, address, timeout=20, **kwargs):
        """
        Initialises the oscilloscope.

        Args:
            name (str): Name of the instrument used by QCoDeS
        address (string): Instrument address as used by VISA
            timeout (float): visa timeout, in secs.
        """

        super().__init__(name, address, timeout=timeout,
                         terminator='\n', **kwargs)
        self.connect_message()
        self.write('*RST')  # resets oscilloscope - initialize to known state
        self.write('*CLS')  # clears status registers and output queue

        # Turns off system headers to allow faster throughput
        # and immediate access to the data values requested by queries.
        self.write(':SYSTem:HEADer OFF')

        # Scope trace boolean
        self.trace_ready = False

        # NOTE: The list of implemented parameters is not complete. Refer to
              # the manual (Infiniium prog guide).

        # Timebase
        self.add_parameter('timebase_reference',
                           label='Reference of the time axis',
                           get_cmd=':TIMebase:REFerence?',
                           set_cmd=':TIMebase:REFerence {}',
                           vals=Enum('CENT', 'CENTer') # NOTE: needs more values
                           )
        self.add_parameter('timebase_range',
                           label='Range of the time axis',
                           unit='s',
                           get_cmd=':TIMebase:RANGe?',
                           set_cmd=':TIMebase:RANGe {}',
                           vals=vals.Numbers(2e-9, 20),
                           get_parser=float,
                           )
        self.add_parameter('timebase_position',
                           label='Offset of the time axis',
                           unit='s',
                           get_cmd=':TIMebase:POSition?',
                           set_cmd=':TIMebase:POSition {}',
                           vals=vals.Numbers(),
                           get_parser=float,
                           )

        # Trigger
        # TODO: add enum for case insesitivity
        self.add_parameter('trigger_edge_source',
                           label='Source channel for the edge trigger',
                           get_cmd=':TRIGger:EDGE:SOURce?',
                           set_cmd=':TRIGger:EDGE:SOURce {}',
                           vals=Enum(*(
                               ['CHANnel{}'.format(i) for i in range(1, 4 + 1)] +
                               ['CHAN{}'.format(i) for i in range(1, 4 + 1)] +
                               ['DIGital{}'.format(i) for i in range(16 + 1)] +
                               ['DIG{}'.format(i) for i in range(16 + 1)] +
                               ['AUX', 'LINE']))
                           )

        self.add_parameter('trigger_edge_slope',
                           label='slope of the edge trigger',
                           get_cmd=':TRIGger:EDGE:SLOPe?',
                           set_cmd=':TRIGger:EDGE:SLOPe {}',
                           vals=Enum('positive', 'negative', 'neither')
                           )

        self.add_parameter('trigger_level_aux',
                           label='Tirgger level AUX',
                           unit='V',
                           get_cmd=':TRIGger:LEVel? AUX',
                           set_cmd=':TRIGger:LEVel AUX,{}',
                           get_parser=float,
                           vals=Numbers(),
                           )

        # Acquisition
        # If sample points, rate and timebase_scale are set in an
        # incommensurate way, the scope only displays part of the waveform

        self.add_parameter('acquire_mode',
                            label='acquisition mode',
                            get_cmd= 'ACQuire:MODE?',
                            set_cmd='ACQuire:MODE {}',
                            vals=Enum('RTIMe', 'ETIMe', 'PDETect',
                                      'HRESolution', 'SEGMented')
                           )

        self.add_parameter('acquire_average',
                           label='average on/off',
                           get_cmd='ACQuire:AVERage?',
                           set_cmd='ACQuire:AVERage {}',
                           val_mapping={True: 1, False: 0},
                           )

        self.add_parameter('acquire_points',
                           label='sample points',
                           get_cmd='ACQuire:POINts?',
                           get_parser=int,
                           set_cmd=self._cmd_and_invalidate('ACQuire:POINts {}'),
                           unit='pts',
                           vals=vals.Numbers(min_value=1, max_value=100e6)
                           )

        self.add_parameter('acquire_sample_rate',
                           label='sample rate',
                           get_cmd='ACQ:SRAT?',
                           set_cmd=self._cmd_and_invalidate('ACQ:SRAT {}'),
                           unit='Sa/s',
                           get_parser=float
                           )

        self.add_parameter('data_source',
                           label='Waveform Data source',
                           get_cmd=':WAVeform:SOURce?',
                           set_cmd=':WAVeform:SOURce {}',
                           vals = Enum( *(
                                ['CHANnel{}'.format(i) for i in range(1, 4+1)]+
                                ['CHAN{}'.format(i) for i in range(1, 4+1)]+
                                ['FUNCtion{}'.format(i) for i in range(1, 16+1)]+
                                ['FUNC{}'.format(i) for i in range(1, 16+1)]+
                                ['WMEMory{}'.format(i) for i in range(1, 4+1)]+
                                ['WMEM{}'.format(i) for i in range(1, 4+1)]+
                                ['HISTogram', 'HIST', 'POD1', 'POD2', 'PODALL']
                           ))
                           )

        ## TODO: implement as array parameter to allow for setting the other filter
        # Ratios
        self.add_parameter('acquire_interpolate',
                            get_cmd=':ACQuire:INTerpolate?',
                            set_cmd=self._cmd_and_invalidate(':ACQuire:INTerpolate {}'),
                            val_mapping={True: 1, False: 0}
                            )

        self.add_parameter('acquire_timespan',
                            get_cmd=(lambda: self.acquire_points.get_latest()
                                            /self.acquire_sample_rate.get_latest()),
                            unit='s',
                            get_parser=float
                            )

        # Time of the first point
        self.add_parameter('waveform_xorigin',
                            get_cmd='WAVeform:XORigin?',
                            unit='s',
                            get_parser=float
                            )

        # Channels
        channels = ChannelList(self, "Channels", InfiniiumChannel,
                                snapshotable=False)

        for i in range(1,5):
            channel = InfiniiumChannel(self, 'chan{}'.format(i), i)
            channels.append(channel)
            self.add_submodule('ch{}'.format(i), channel)
        channels.lock()
        self.add_submodule('channels', channels)


    def _cmd_and_invalidate(self, cmd: str) -> Callable:
        return partial(Infiniium._cmd_and_invalidate_call, self, cmd)

    def _cmd_and_invalidate_call(self, cmd: str, val) -> None:
        """
        Executes command and sets trace_ready status to false
        Any command that effects the number of setpoints should invalidate the trace
        """
        self.trace_ready = False
        self.write(cmd.format(val))


class InfiniiumChannel(InstrumentChannel):

    def __init__(self, parent, name, channel):
        super().__init__(parent, name)

        self.add_parameter(name='display',
                           label='Channel {} display on/off'.format(channel),
                           set_cmd='CHANnel{}:DISPlay {{}}'.format(channel),
                           get_cmd='CHANnel{}:DISPlay?'.format(channel),
                           val_mapping={True: 1, False: 0},
                           )

        self.add_parameter(name='offset',
                           label='Channel {} offset'.format(channel),
                           unit='V',
                           set_cmd='CHANnel{}:OFFSet {{}}'.format(channel),
                           get_cmd='CHANnel{}:OFFSet?'.format(channel),
                           get_parser=float,
                           vals=vals.Numbers()
                           )

        self.add_parameter(name='range',
                           label='Channel {} range'.format(channel),
                           unit='V',
                           set_cmd='CHANnel{}:RANGe {{}}'.format(channel),
                           get_cmd='CHANnel{}:RANGe?'.format(channel),
                           get_parser=float,
                           vals=vals.Numbers()
                           )

        self.add_parameter('trigger_level',
                           label='Trigger level channel {}'.format(channel),
                           unit='V',
                           get_cmd=':TRIGger:LEVel? CHANnel{}'.format(channel),
                           set_cmd=':TRIGger:LEVel CHANnel{},{{}}'.format(channel),
                           get_parser=float,
                           vals=vals.Numbers()
                           )

        # Acquisition
        self.add_parameter(name='trace',
                           channel=channel,
                           parameter_class=RawTrace
                           )


class RawTrace(ArrayParameter):
    """
    Returns a data trace from oscilloscope
    """

    def __init__(self, name, parent, channel):
        super().__init__(name,
                         shape=(1024,),
                         label='Voltage',
                         unit='V',
                         setpoint_names=('Time',),
                         setpoint_labels=(
                             'Channel {} time series'.format(channel),),
                         setpoint_units=('s',),
                         docstring='raw trace from the scope',
                         )
        self._channel = channel
        self._instrument = parent

    def prepare_curvedata(self):
        """
        Prepare the scope for returning curve data

        To calculate set points, we must have the full preamble
        For the instrument to return the full preamble, the channel
        in question must be displayed
        """

        instr = self._instrument
        instr.write(':DIGitize CHANnel{}'.format(self._channel))
        instr.write(':CHANnel{}:DISPlay ON'.format(self._channel))

        # number of set points
        self.npts = int(instr.ask("WAVeform:POINts?"))
        # first set point
        self.xorigin = float(instr.ask(":WAVeform:XORigin?"))
        # step size
        self.xincrem = float(instr.ask(":WAVeform:XINCrement?"))
        # calculate set points
        xdata = np.linspace(self.xorigin,
                            self.npts * self.xincrem + self.xorigin, self.npts)

        # set setpoints
        self.setpoints = (tuple(xdata),)
        self.shape = (self.npts, )

        self._instrument._parent.trace_ready = True

    def get(self):
        # When get is called, the setpoints have to be known already (saving data issue).
        # Therefore run prepare first function that queries for the size.
        # Checks if it is already prepared

        instr = self._instrument

        if not instr._parent.trace_ready:
            raise TraceNotReady('Please run prepare_curvedata to prepare '
                                'the scope for acquiring a trace.')

        # TODO: check number of points

        # realtime mode: only one trigger is used
        instr._parent.acquire_mode('RTIMe')

        # digitize is the actual call for acquisition, blocks
        instr.write(':DIGitize CHANnel{}'.format(self._channel))

        # select the channel from which to read
        instr._parent.data_source('CHAN{}'.format(self._channel))

        # specify the data format in which to read
        instr.write(':WAVeform:FORMat WORD')
        instr.write(":WAVeform:BYTeorder LSBFirst")

        # request the actual transfer
        data = instr._parent.visa_handle.query_binary_values(
            'WAV:DATA?', datatype='h', is_big_endian=False)
        if len(data) != self.shape[0]:
            print('{} points have been acquired and {} \
            set points have been prepared in \
            prepare_curvedata'.format(len(data), self.shape[0]))

        # check x data scaling
        xorigin = float(instr.ask(":WAVeform:XORigin?"))
        xincrem = float(instr.ask(":WAVeform:XINCrement?"))
        error = self.xorigin - xorigin
        # this is a bad workaround
        if error > xincrem:
            print('{} is the prepared x origin and {} \
            is the x origin after the measurement.'.format(self.xorigin,
                                                           xorigin))
        error = (self.xincrem - xincrem) / xincrem
        if error > 1e-6:
            print('{} is the prepared x increment and {} \
            is the x increment after the measurement.'.format(self.xincrem,
                                                              xincrem))

        # y data scaling
        yorigin = float(instr.ask(":WAVeform:YORigin?"))
        yinc = float(instr.ask(":WAVeform:YINCrement?"))
        channel_data = np.array(data)
        channel_data = np.multiply(channel_data, yinc) + yorigin

        # restore original state
        instr.write(':CHANnel{}:DISPlay ON'.format(self._channel))
        # continue refresh
        instr.write(':RUN')

        return channel_data


class TraceNotReady(Exception):
    pass


class TraceSetPointsChanged(Exception):
    pass