import logging
from typing import Dict, Callable
from functools import partial

import numpy as np

from qcodes import VisaInstrument, validators as vals
from qcodes import InstrumentChannel, ChannelList, Instrument
from qcodes import ArrayParameter
from qcodes.utils.validators import Enum, Numbers


log = logging.getLogger(__name__)


class TraceNotReady(Exception):
    pass


class TraceSetPointsChanged(Exception):
    pass


class RawTrace(ArrayParameter):
    """
    raw_trace will return a trace from OSCIL
    """

    def __init__(self, name, instrument, channel):
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
        self._instrument = instrument

    def prepare_curvedata(self):
        """
        Prepare the scope for returning curve data
        """
        # To calculate set points, we must have the full preamble
        # For the instrument to return the full preamble, the channel
        # in question must be displayed

        # shorthand
        instr = self._instrument
        # number of set points
        self.npts = int(instr.ask("WAV:POIN?"))
        # first set point
        self.xorigin = float(instr.ask(":WAVeform:XORigin?"))
        # step size
        self.xincrem = float(instr.ask(":WAVeform:XINCrement?"))
        # calculate set points
        xdata = np.linspace(self.xorigin,
                            self.npts * self.xincrem + self.xorigin, self.npts)

        # set setpoints
        self.setpoints = (tuple(xdata), )
        self.shape = (self.npts, )

        # make this on a per channel basis?
        self._instrument._parent.trace_ready = True

    def get(self):
        # when get is called the setpoints have to be known already
        # (saving data issue). Therefor create additional prepare function that
        # queries for the size.
        # check if already prepared
        if not self._instrument._parent.trace_ready:
            raise TraceNotReady('Please run prepare_curvedata to prepare '
                                'the scope for acquiring a trace.')

        # shorthand
        instr = self._instrument

        # set up the instrument
        # ---------------------------------------------------------------------

        # TODO: check number of points
        # check if requested number of points is less than 500 million

        # get intrument state
        state = instr.ask(':RSTate?')
        # realtime mode: only one trigger is used
        instr._parent.acquire_mode('RTIMe')

        # acquire the data
        # ---------------------------------------------------------------------

        # digitize is the actual call for acquisition, blocks
        instr.write(':DIGitize CHANnel{}'.format(self._channel))

        # transfer the data
        # ---------------------------------------------------------------------

        # select the channel from which to read
        instr._parent.data_source('CHAN{}'.format(self._channel))
        # specifiy the data format in which to read
        instr.write(':WAVeform:FORMat WORD')
        instr.write(":waveform:byteorder LSBFirst")
        # streaming is only required for data > 1GB
        instr.write(':WAVeform:STReaming OFF')

        # request the actual transfer
        data = instr._parent.visa_handle.query_binary_values(
            'WAV:DATA?', datatype='h', is_big_endian=False,
            expect_termination=False)
        # the Infiniium does not include an extra termination char on binary
        # messages so we set expect_termination to False

        if len(data) != self.shape[0]:
            raise TraceSetPointsChanged('{} points have been aquired and {} \
            set points have been prepared in \
            prepare_curvedata'.format(len(data), self.shape[0]))
        # check x data scaling
        xorigin = float(instr.ask(":WAVeform:XORigin?"))
        # step size
        xincrem = float(instr.ask(":WAVeform:XINCrement?"))
        error = self.xorigin - xorigin
        # this is a bad workaround
        if error > xincrem:
            raise TraceSetPointsChanged('{} is the prepared x origin and {} \
            is the x origin after the measurement.'.format(self.xorigin,
                                                           xorigin))
        error = (self.xincrem - xincrem) / xincrem
        if error > 1e-6:
            raise TraceSetPointsChanged('{} is the prepared x increment and {} \
            is the x increment after the measurement.'.format(self.xincrem,
                                                              xincrem))
        # y data scaling
        yorigin = float(instr.ask(":WAVeform:YORigin?"))
        yinc = float(instr.ask(":WAVeform:YINCrement?"))
        channel_data = np.array(data)
        channel_data = np.multiply(channel_data, yinc) + yorigin

        # restore original state
        # ---------------------------------------------------------------------

        # switch display back on
        instr.write(':CHANnel{}:DISPlay ON'.format(self._channel))
        # continue refresh
        if state == 'RUN':
            instr.write(':RUN')

        return channel_data


class MeasurementSubsystem(InstrumentChannel):
    """
    Submodule containing the measurement subsystem commands and associated
    parameters
    """
    # note: this is not really a channel, but InstrumentChannel does everything
    # a 'Submodule' class should do

    def __init__(self, parent: Instrument, name: str, **kwargs) -> None:
        super().__init__(parent, name, **kwargs)

        self.add_parameter(name='source_1',
                           label='Measurement primary source',
                           set_cmd=partial(self._set_source, 1),
                           get_cmd=partial(self._get_source, 1),
                           val_mapping={i: f'CHAN{i}' for i in range(1, 5)},
                           snapshot_value=False)

        self.add_parameter(name='source_2',
                           label='Measurement secondary source',
                           set_cmd=partial(self._set_source, 2),
                           get_cmd=partial(self._get_source, 2),
                           val_mapping={i: f'CHAN{i}' for i in range(1, 5)},
                           snapshot_value=False)

        self.add_parameter(name='amplitude',
                           label='Voltage amplitude',
                           get_cmd=self._make_meas_cmd('VAMPlitude'),
                           get_parser=float,
                           unit='V',
                           snapshot_value=False)

        self.add_parameter(name='average',
                           label='Voltage average',
                           get_cmd=self._make_meas_cmd('VAVerage'),
                           get_parser=float,
                           unit='V',
                           snapshot_value=False)

        self.add_parameter(name='base',
                           label='Statistical base',
                           get_cmd=self._make_meas_cmd('VBASe'),
                           get_parser=float,
                           unit='V',
                           snapshot_value=False)

        self.add_parameter(name='frequency',
                           label='Signal frequency',
                           get_cmd=self._make_meas_cmd('FREQuency'),
                           get_parser=float,
                           unit='Hz',
                           docstring="""
                                     measure the frequency of the first
                                     complete cycle on the screen using
                                     the mid-threshold levels of the waveform
                                     """,
                           snapshot_value=False)

        self.add_parameter(name='lower',
                           label='Voltage lower',
                           get_cmd=self._make_meas_cmd('VLOWer'),
                           get_parser=float,
                           unit='V',
                           snapshot_value=False)

        self.add_parameter(name='max',
                           label='Voltage maximum',
                           get_cmd=self._make_meas_cmd('VMAX'),
                           get_parser=float,
                           unit='V',
                           snapshot_value=False)

        self.add_parameter(name='middle',
                           label='Middle threshold voltage',
                           get_cmd=self._make_meas_cmd('VMIDdle'),
                           get_parser=float,
                           unit='V',
                           snapshot_value=False)

        self.add_parameter(name='min',
                           label='Voltage minimum',
                           get_cmd=self._make_meas_cmd('VMIN'),
                           get_parser=float,
                           unit='V',
                           snapshot_value=False)

        self.add_parameter(name='overshoot',
                           label='Voltage overshoot',
                           get_cmd=self._make_meas_cmd('VOVershoot'),
                           get_parser=float,
                           unit='V',
                           snapshot_value=False)

        self.add_parameter(name='vpp',
                           label='Voltage peak-to-peak',
                           get_cmd=self._make_meas_cmd('VPP'),
                           get_parser=float,
                           unit='V',
                           snapshot_value=False)

        self.add_parameter(name='rms',
                           label='Voltage RMS',
                           get_cmd=self._make_meas_cmd('VRMS') + ' DISPlay, DC',
                           get_parser=float,
                           unit='V',
                           snapshot_value=False)

        self.add_parameter(name='rms_no_DC',
                           label='Voltage RMS',
                           get_cmd=self._make_meas_cmd('VRMS') + ' DISPlay, AC',
                           get_parser=float,
                           unit='V',
                           snapshot_value=False)

    @staticmethod
    def _make_meas_cmd(cmd: str) -> str:
        """
        Helper function to avoid typos
        """
        return f':MEASure:{cmd}?'

    def _set_source(self, rank: int, source: str) -> None:
        """
        Set the measurement source, either primary (rank==1) or secondary
        (rank==2)
        """
        sources = self.ask(':MEASure:SOURCE?').split(',')
        if rank == 1:
            self.write(f':MEASure:SOURCE {source}, {sources[1]}')
        else:
            self.write(f':MEASure:SOURCE {sources[0]}, {source}')

    def _get_source(self, rank: int) -> str:
        """
        Get the measurement source, either primary (rank==1) or secondary
        (rank==2)
        """
        sources = self.ask(':MEASure:SOURCE?').split(',')

        return sources[rank-1]


class InfiniiumChannel(InstrumentChannel):

    def __init__(self, parent, name, channel):
        super().__init__(parent, name)
        # display
        self.add_parameter(name='display',
                           label='Channel {} display on/off'.format(channel),
                           set_cmd='CHANnel{}:DISPlay {{}}'.format(channel),
                           get_cmd='CHANnel{}:DISPlay?'.format(channel),
                           val_mapping={True: 1, False: 0},
                           )
        # scaling
        self.add_parameter(name='offset',
                           label='Channel {} offset'.format(channel),
                           set_cmd='CHAN{}:OFFS {{}}'.format(channel),
                           unit='V',
                           get_cmd='CHAN{}:OFFS?'.format(channel),
                           get_parser=float
                           )

        # scale and range are interdependent, when setting one, invalidate the
        # the other.
        # Scale is commented out for this reason
        # self.add_parameter(name='scale',
        #                    label='Channel {} scale'.format(channel),
        #                    unit='V/div',
        #                    set_cmd='CHAN{}:SCAL {{}}'.format(channel),
        #                    get_cmd='CHAN{}:SCAL?'.format(channel),
        #                    get_parser=float,
        #                    vals=vals.Numbers(0,100)  # TODO: upper limit?
        #                    )

        self.add_parameter(name='range',
                           label='Channel {} range'.format(channel),
                           unit='V',
                           set_cmd='CHAN{}:RANG {{}}'.format(channel),
                           get_cmd='CHAN{}:RANG?'.format(channel),
                           get_parser=float,
                           vals=vals.Numbers()
                           )
        # trigger
        self.add_parameter(
            'trigger_level',
            label='Tirgger level channel {}'.format(channel),
            unit='V',
            get_cmd=':TRIGger:LEVel? CHANnel{}'.format(channel),
            set_cmd=':TRIGger:LEVel CHANnel{},{{}}'.format(channel),
            get_parser=float,
            vals=Numbers(),
        )

        # Acquisition
        self.add_parameter(name='trace',
                           channel=channel,
                           parameter_class=RawTrace
                           )


class Infiniium(VisaInstrument):
    """
    This is the QCoDeS driver for the Keysight Infiniium oscilloscopes from the
     - tested for MSOS104A of the Infiniium S-series.
    """

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

        # Scope trace boolean
        self.trace_ready = False

        # switch the response header off,
        # else none of our parameters will work
        self.write(':SYSTem:HEADer OFF')

        # functions

        # general parameters

        # the parameters are in the same order as the front panel.
        # Beware, he list of implemented parameters is not complete. Refer to
        # the manual (Infiniium prog guide) for an equally infiniium list.

        # time base

        # timebase_scale is commented out for same reason as channel scale
        # use range instead
        # self.add_parameter('timebase_scale',
        #                    label='Scale of the one time devision',
        #                    unit='s/Div',
        #                    get_cmd=':TIMebase:SCALe?',
        #                    set_cmd=':TIMebase:SCALe {}',
        #                    vals=Numbers(),
        #                    get_parser=float,
        #                    )

        self.add_parameter('timebase_range',
                           label='Range of the time axis',
                           unit='s',
                           get_cmd=':TIMebase:RANGe?',
                           set_cmd=':TIMebase:RANGe {}',
                           vals=Numbers(5e-12, 20),
                           get_parser=float,
                           )
        self.add_parameter('timebase_position',
                           label='Offset of the time axis',
                           unit='s',
                           get_cmd=':TIMebase:POSition?',
                           set_cmd=':TIMebase:POSition {}',
                           vals=Numbers(),
                           get_parser=float,
                           )

        self.add_parameter('timebase_roll_enabled',
                           label='Is rolling mode enabled',
                           get_cmd=':TIMebase:ROLL:ENABLE?',
                           set_cmd=':TIMebase:ROLL:ENABLE {}',
                           val_mapping={True: 1, False: 0}
                           )

        # trigger
        self.add_parameter('trigger_enabled',
                           label='Is trigger enabled',
                           get_cmd=':TRIGger:AND:ENABLe?',
                           set_cmd=':TRIGger:AND:ENABLe {}',
                           val_mapping={True: 1, False: 0}
                           )

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
                           )  # add enum for case insesitivity
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
        # Aquisition
        # If sample points, rate and timebase_scale are set in an
        # incomensurate way, the scope only displays part of the waveform
        self.add_parameter('acquire_points',
                           label='sample points',
                           get_cmd='ACQ:POIN?',
                           get_parser=int,
                           set_cmd=self._cmd_and_invalidate('ACQ:POIN {}'),
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

        # this parameter gets used internally for data aquisition. For now it
        # should not be used manually
        self.add_parameter('data_source',
                           label='Waveform Data source',
                           get_cmd=':WAVeform:SOURce?',
                           set_cmd=':WAVeform:SOURce {}',
                           vals = Enum( *(\
                                ['CHANnel{}'.format(i) for i in range(1, 4+1)]+\
                                ['CHAN{}'.format(i) for i in range(1, 4+1)]+\
                                ['DIFF{}'.format(i) for i in range(1, 2+1)]+\
                                ['COMMonmode{}'.format(i) for i in range(3, 4+1)]+\
                                ['COMM{}'.format(i) for i in range(3, 4+1)]+\
                                ['FUNCtion{}'.format(i) for i in range(1, 16+1)]+\
                                ['FUNC{}'.format(i) for i in range(1, 16+1)]+\
                                ['WMEMory{}'.format(i) for i in range(1, 4+1)]+\
                                ['WMEM{}'.format(i) for i in range(1, 4+1)]+\
                                ['BUS{}'.format(i) for i in range(1, 4+1)]+\
                                ['HISTogram', 'HIST', 'CLOCK']+\
                                ['MTRend', 'MTR']))
                           )

        # TODO: implement as array parameter to allow for setting the other filter
        # ratios
        self.add_parameter('acquire_interpolate',
                            get_cmd=':ACQuire:INTerpolate?',
                            set_cmd=self._cmd_and_invalidate(':ACQuire:INTerpolate {}'),
                            val_mapping={True: 1, False: 0}
                            )

        self.add_parameter('acquire_mode',
                            label='Acquisition mode',
                            get_cmd= 'ACQuire:MODE?',
                            set_cmd='ACQuire:MODE {}',
                            vals=Enum('ETIMe', 'RTIMe', 'PDETect',
                                      'HRESolution', 'SEGMented',
                                      'SEGPdetect', 'SEGHres')
                            )

        self.add_parameter('acquire_timespan',
                            get_cmd=(lambda: self.acquire_points.get_latest() \
                                            /self.acquire_sample_rate.get_latest()),
                            unit='s',
                            get_parser=float
                            )

        # time of the first point
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

        # Submodules
        meassubsys = MeasurementSubsystem(self, 'measure')
        self.add_submodule('measure', meassubsys)

    def _cmd_and_invalidate(self, cmd: str) -> Callable:
        return partial(Infiniium._cmd_and_invalidate_call, self, cmd)

    def _cmd_and_invalidate_call(self, cmd: str, val) -> None:
        """
        executes command and sets trace_ready status to false
        any command that effects the number of setpoints should invalidate the trace
        """
        self.trace_ready = False
        self.write(cmd.format(val))
