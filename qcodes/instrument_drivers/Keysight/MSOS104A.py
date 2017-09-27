import logging
import binascii
from typing import List, Dict

import numpy as np
from pyvisa.errors import VisaIOError

from qcodes import VisaInstrument, validators as vals
from qcodes import InstrumentChannel, ChannelList
from qcodes import ArrayParameter
from qcodes.utils.validators import Enum, Numbers


log = logging.getLogger(__name__)


class TraceNotReady(Exception):
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
                         setpoint_labels=( \
                                'Channel {} time series'.format(channel),),
                         setpoint_units=('s',),
                         docstring='raw trace from the scope',
                         )
        self._channel = channel
        self._instrument = instrument


    def set_sweep(self, start, stop, npts):
        t = tuple(np.linspace(float(start), float(stop), num=npts))
        self.setpoints = (t,)
        self.shape = (npts,)


    def get(self):
        #self._instrument.write('DIGITIZE')
        # oscil = self._instrument._parent.visa_handle
        # the instrument should be used so self instead of oscil.
        # ask and write are implemented, and they do exception handling
        # and mocking
        # also the parent is wrapped by the channel as a parent instrument
        
        # shorthand
        oscil=self._instrument

        # set up the instrument
        # ---------------------------------------------------------------------

        # check if requested number of points is less than 500 million

        # realtime mode: only one trigger is used
        instr._parent.acquire_mode('RTIMe')
        
        # acquire the data
        # ---------------------------------------------------------------------

        # digitize is the actual call for acquisition, blocks
        oscil.write(':DIGitize CHANnel{}'.format(self._channel))

        # switch display back on
        oscil.write(':CHANnel1:DISPlay ON')

        # transfer the data
        # ---------------------------------------------------------------------

        # switch the response header off for lower overhead
        oscil.write(':SYSTem:HEADer OFF')
        # select the channel from which to read
        self._instrument._parent.data_source('CHAN{}'.format(self._channel))
        # specifiy the data format in which to read
        oscil.write(':WAVeform:FORMat WORD')
        # streaming is only required for data > 1GB
        oscil.write(':WAVeform:STReaming OFF')
        # request the actual transfer
        oscil.write(':WAVeform:DATA?')

        visa_handle = oscil._parent.visa_handle
        # first read the header
        data = visa_handle.read_raw(2)

        # then read the remaining data

        # check if all data was received



        npts = float(oscil.ask("WAV:POIN?"))
        xorigin = float(oscil.ask(":WAVeform:XORigin?"))
        yorigin = float(oscil.ask(":WAVeform:YORigin?"))
        yinc = float(oscil.ask(":WAVeform:YINCrement?"))


        srate = self.aquire_sample_rate.get_latest()

        timestepsize = 1/srate
        start = xorigin
        stop = timestepsize*npts+xorigin
        # what is the correct number of points?
        npts = self.acquire_points.get_latest()

        #
        self.shape = (npts,)
        self.set_sweep(start, stop, npts)
        #TODO: fix above set when any of the above things change

        # temporarily remove the termination string
        # this should be done with the termination context manager
        term = oscil.read_termination
        oscil.read_termination = None
        #self._instrument.visa_handle.end_input = False
        # this should be bytes
        nbits = int(2*npts +3)

        oscil.write(":waveform:data?")
        # set waveform streaming on to get data as response
        data = oscil.read_raw(nbits)

        channel_data = array.array('h')
        channel_data.fromstring(data[2:-1]) #skip first 2 bytes and last byte
        #channel_data.byteswap()
        #channel_data.reverse()
        channel_data = np.array(channel_data)
        channel_data = np.multiply(channel_data, yinc) + yorigin

        oscil.read_termination = term

        return channel_data

    def _setup_instrument_for_acquisition(self):
        instr = self._instrument
        instr.write(":waveform:format WORD")
        instr.write(":waveform:byteorder LSBFirst")
        instr._parent.acquire_mode('RTIMe')
        # instr.write(":waveform:streaming 1")

class MSOChannel(InstrumentChannel):

    def __init__(self, parent, name, channel):
        super().__init__(parent, name)
        # scaling
        self.add_parameter(name='offset',
                           label='Channel {} offset'.format(channel),
                           set_cmd='CHAN{}:OFFS {{}}'.format(channel),
                           unit='V',
                           get_cmd='CHAN{}:OFFS?'.format(channel),
                           get_parser=float
                           )

        self.add_parameter(name='scale',
                           label='Channel {} scale'.format(channel),
                           unit='V/div',
                           set_cmd='CHAN{}:SCAL {{}}'.format(channel),
                           get_cmd='CHAN{}:SCAL?'.format(channel),
                           get_parser=float,
                           vals=vals.Numbers(0,100)  # TODO: upper limit?
                           )

        self.add_parameter(name='range',
                           label='Channel {} range'.format(channel),
                           unit='V',
                           set_cmd='CHAN{}:RANG {{}}'.format(channel),
                           get_cmd='CHAN{}:RANG?'.format(channel),
                           get_parser=float,
                           vals=vals.Numbers()
                           )
        # trigger
        self.add_parameter('trigger_level',
                           label='Tirgger level channel {}'.format(channel),
                           unit='V',
                           get_cmd=':TRIGger:LEVel? CHANnel{}'.format(channel),
                           set_cmd=\
                               ':TRIGger:LEVel CHANnel{},{{}}'.format(channel),
                           get_parser=float,
                           vals=Numbers(),
                          )

        # Acquisition
        self.add_parameter( name='trace',
                            channel = channel,
                            parameter_class=RawTrace
                            )
        def snapshot_base(self, update: bool=False) -> Dict:
            params_to_skip_update = ['trace']
            super().snapshot_base(update=update,
                                  params_to_skip_update=params_to_skip_update)


class MSOS104A(VisaInstrument):
    """
    This is the QCoDeS driver for the Keysight MSOS104A oscilloscope from the
    Infiniium S-series.
    """
    def __init__(self, name, address, timeout=20, **kwargs):
        """
        Initialises the MSOS104A.

        Args:
            name (str): Name of the instrument used by QCoDeS
        address (string): Instrument address as used by VISA
            timeout (float): visa timeout, in secs.
        """

        super().__init__(name, address, timeout=timeout,\
                         terminator='\n', **kwargs)
        self.connect_message()

        # functions

        # general parameters

        # the parameters are in the same order as the front panel.
        # Beware, he list of implemented parameters is not complete. Refer to
        # the manual (Infiniium prog guide) for an equally infiniium list.

        # time base
        self.add_parameter('timebase_scale',
                           label='Scale of the time axis',
                           unit='s',
                           get_cmd=':TIMebase:SCALe?',
                           set_cmd=':TIMebase:SCALe {}',
                           vals=Numbers(),
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
                           vals = Enum( *(\
                                ['CHANnel{}'.format(i) for i in range(1,4+1)]+\
                                ['CHAN{}'.format(i) for i in range(1,4+1)]+\
                                ['DIGital{}'.format(i) for i in range(16+1)]+\
                                ['DIG{}'.format(i) for i in range(16+1)]+\
                                ['AUX', 'LINE']))
                          )# add enum for case insesitivity
        self.add_parameter('trigger_edge_slope',
                           label='slope of the edge trigger',
                           get_cmd=':TRIGger:EDGE:SLOPe?',
                           set_cmd=':TRIGger:EDGE:SLOPe {}',
                           vals = Enum('positive', 'negative', 'neither')
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
        #TODO: check what these points are
        self.add_parameter('acquire_points',
                           label='sample points',
                           get_cmd='ACQ:POIN?',
                           get_parser=int,
                           #set_cmd=self._set_points,
                           set_cmd='ACQ:POIN {}',
                           unit='pts',
                           vals=vals.Numbers(min_value=1, max_value=100e6)
                           )

        self.add_parameter('acquire_sample_rate',
                            label='sample rate',
                            get_cmd= 'ACQ:SRAT?',
                            set_cmd='ACQ:SRAT {}',
                            unit='Sa/s',
                            get_parser=float
                            )
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
                            set_cmd=':ACQuire:INTerpolate {}',
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
        channels = ChannelList(self, "Channels", MSOChannel,
                                snapshotable=False)

        for i in range(1,5):
            channel = MSOChannel(self, 'chan{}'.format(i), i)
            channels.append(channel)
            self.add_submodule('ch{}'.format(i), channel)
        channels.lock()
        self.add_submodule('channels', channels)



# compatitbility/setup
# change of names with acquire_ prefix
# add acquire_interpolate(1) to the setup
