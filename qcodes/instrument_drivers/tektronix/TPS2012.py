import logging
import binascii

import numpy as np
from pyvisa.errors import VisaIOError
from functools import partial

from qcodes import VisaInstrument, validators as vals
from qcodes import InstrumentChannel, ChannelList
from qcodes import ArrayParameter

log = logging.getLogger(__name__)


class TraceNotReady(Exception):
    pass


class ScopeArray(ArrayParameter):
    def __init__(self, name, instrument, channel):
        super().__init__(name=name,
                         shape=(2500,),
                         label='Voltage',
                         unit='V ',
                         setpoint_names=('Time', ),
                         setpoint_labels=('Time', ),
                         setpoint_units=('s',),
                         docstring='holds an array from scope')
        self.channel = channel
        self._instrument = instrument

    def calc_set_points(self):
        message = self._instrument.ask('WFMPre?')
        preamble = self._preambleparser(message)
        xstart = preamble['x_zero']
        xinc = preamble['x_incr']
        no_of_points = preamble['no_of_points']
        xdata = np.linspace(xstart, no_of_points * xinc + xstart, no_of_points)
        return xdata, no_of_points

    def prepare_curvedata(self):
        """
        Prepare the scope for returning curve data
        """
        # To calculate set points, we must have the full preamble
        # For the instrument to return the full preamble, the channel
        # in question must be displayed

        self._instrument.parameters['state'].set('ON')
        self._instrument._parent.data_source('CH{}'.format(self.channel))

        xdata, no_of_points = self.calc_set_points()
        self.setpoints = (tuple(xdata), )
        self.shape = (no_of_points, )

        self._instrument._parent.trace_ready = True

    def get_raw(self):
        if not self._instrument._parent.trace_ready:
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

    def _curveasker(self, ch):
        self._instrument.write('DATa:SOURce CH{}'.format(ch))
        message = self._instrument.ask('WAVFrm?')
        self._instrument.write('*WAI')
        return message

    @staticmethod
    def _binaryparser(curve):
        """
        Helper function for parsing the curve data

        Args:
            curve (str): the return value of 'CURVe?' when
              DATa:ENCdg is set to RPBinary.
              Note: The header and final newline character
              must be removed.

        Returns:
            nparray: the curve in units where the digitisation range
              is mapped to (-32768, 32767).
        """
        # TODO: Add support for data width = 1 mode?
        output = np.zeros(int(len(curve)/2))  # data width 2
        # output = np.zeros(int(len(curve)))  # data width 1
        for ii, _ in enumerate(output):
            # casting FTWs
            temp = curve[2*ii:2*ii+1].encode('latin-1')  # data width 2
            temp = binascii.b2a_hex(temp)
            temp = (int(temp, 16)-128)*256  # data width 2 (1)
            output[ii] = temp
        return output

    @staticmethod
    def _preambleparser(response):
        """
        Parser function for the curve preamble

        Args:
            response (str): The response of WFMPre?

        Returns:
            dict: a dictionary containing the following keys:
              no_of_bytes, no_of_bits, encoding, binary_format,
              byte_order, no_of_points, waveform_ID, point_format,
              x_incr, x_zero, x_unit, y_multiplier, y_zero, y_offset, y_unit
        """
        response = response.split(';')
        outdict = {}
        outdict['no_of_bytes'] = int(response[0])
        outdict['no_of_bits'] = int(response[1])
        outdict['encoding'] = response[2]
        outdict['binary_format'] = response[3]
        outdict['byte_order'] = response[4]
        outdict['no_of_points'] = int(response[5])
        outdict['waveform_ID'] = response[6]
        outdict['point_format'] = response[7]
        outdict['x_incr'] = float(response[8])
        # outdict['point_offset'] = response[9]  # Always zero
        outdict['x_zero'] = float(response[10])
        outdict['x_unit'] = response[11]
        outdict['y_multiplier'] = float(response[12])
        outdict['y_zero'] = float(response[13])
        outdict['y_offset'] = float(response[14])
        outdict['y_unit'] = response[15]

        return outdict

    def _curveparameterparser(self, waveform):
        """
        The parser for the curve parameter. Note that WAVFrm? is equivalent
        to WFMPre?; CURVe?

        Args:
            waveform (str): The return value of WAVFrm?

        Returns:
            (np.array, np.array): Two numpy arrays with the time axis in units
            of s and curve values in units of V; (time, voltages)
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


class TPS2012Channel(InstrumentChannel):

    def __init__(self, parent, name, channel):
        super().__init__(parent, name)

        self.add_parameter('scale',
                           label='Channel {} Scale'.format(channel),
                           unit='V/div',
                           get_cmd='CH{}:SCAle?'.format(channel),
                           set_cmd='CH{}:SCAle {}'.format(channel, '{}'),
                           get_parser=float
                           )
        self.add_parameter('position',
                           label='Channel {} Position'.format(channel),
                           unit='div',
                           get_cmd='CH{}:POSition?'.format(channel),
                           set_cmd='CH{}:POSition {}'.format(channel, '{}'),
                           get_parser=float
                           )
        self.add_parameter('curvedata',
                           channel=channel,
                           parameter_class=ScopeArray,
                           )
        self.add_parameter('state',
                           label='Channel {} display state'.format(channel),
                           set_cmd='SELect:CH{} {}'.format(channel, '{}'),
                           get_cmd=partial(self._get_state, channel),
                           val_mapping={'ON': 1, 'OFF': 0},
                           vals=vals.Enum('ON', 'OFF')
                           )

    def _get_state(self, ch):
        """
        get_cmd for the chX_state parameter
        """
        # 'SELect?' returns a ';'-separated string of 0s and 1s
        # denoting state display state of ch1, ch2, ?, ?, ?
        # (maybe ch1, ch2, math, ref1, ref2 ..?)
        selected = list(map(int, self.ask('SELect?').split(';')))
        state = selected[ch - 1]
        return state


class TPS2012(VisaInstrument):
    """
    This is the QCoDeS driver for the Tektronix 2012B oscilloscope.
    """

    def __init__(self, name, address, timeout=20, **kwargs):
        """
        Initialises the TPS2012.

        Args:
            name (str): Name of the instrument used by QCoDeS
        address (string): Instrument address as used by VISA
            timeout (float): visa timeout, in secs. long default (180)
              to accommodate large waveforms
        """

        super().__init__(name, address, timeout=timeout, **kwargs)
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
        self.add_parameter('trigger_type',
                           label='Type of the trigger',
                           get_cmd='TRIGger:MAIn:TYPe?',
                           set_cmd='TRIGger:MAIn:TYPe {}',
                           vals=vals.Enum('EDGE', 'VIDEO', 'PULSE')
                           )
        self.add_parameter('trigger_source',
                           label='Source for the trigger',
                           get_cmd='TRIGger:MAIn:EDGE:SOURce?',
                           set_cmd='TRIGger:MAIn:EDGE:SOURce {}',
                           vals=vals.Enum('CH1', 'CH2')
                           )
        self.add_parameter('trigger_edge_slope',
                           label='Slope for edge trigger',
                           get_cmd='TRIGger:MAIn:EDGE:SLOpe?',
                           set_cmd='TRIGger:MAIn:EDGE:SLOpe {}',
                           vals=vals.Enum('FALL', 'RISE')
                           )
        self.add_parameter('trigger_level',
                           label='Trigger level',
                           unit='V',
                           get_cmd='TRIGger:MAIn:LEVel?',
                           set_cmd='TRIGger:MAIn:LEVel {}',
                           vals=vals.Numbers()
                           )
        self.add_parameter('data_source',
                           label='Data source',
                           get_cmd='DATa:SOUrce?',
                           set_cmd='DATa:SOURce {}',
                           vals=vals.Enum('CH1', 'CH2')
                           )
        self.add_parameter('horizontal_scale',
                           label='Horizontal scale',
                           unit='s',
                           get_cmd='HORizontal:SCAle?',
                           set_cmd=self._set_timescale,
                           get_parser=float,
                           vals=vals.Enum(5e-9, 10e-9, 25e-9, 50e-9, 100e-9,
                                          250e-9, 500e-9, 1e-6, 2.5e-6, 5e-6,
                                          10e-6, 25e-6, 50e-6, 100e-6, 250e-6,
                                          500e-6, 1e-3, 2.5e-3, 5e-3, 10e-3,
                                          25e-3, 50e-3, 100e-3, 250e-3, 500e-3,
                                          1, 2.5, 5, 10, 25, 50))

        # channel-specific parameters
        channels = ChannelList(self, "ScopeChannels", TPS2012Channel, snapshotable=False)
        for ch_num in range(1, 3):
            ch_name = "ch{}".format(ch_num)
            channel = TPS2012Channel(self, ch_name, ch_num)
            channels.append(channel)
            self.add_submodule(ch_name, channel)
        channels.lock()
        self.add_submodule("channels", channels)

        # Necessary settings for parsing the binary curve data
        self.visa_handle.encoding = 'latin-1'
        log.info('Set VISA encoding to latin-1')
        self.write('DATa:ENCdg RPBinary')
        log.info('Set TPS2012 data encoding to RPBinary' +
                 ' (Positive Integer Binary)')
        self.write('DATa:WIDTh 2')
        log.info('Set TPS2012 data width to 2')
        # Note: using data width 2 has been tested to not add
        # significantly to transfer times. The maximal length
        # of an array in one transfer is 2500 points.

    def _set_timescale(self, scale):
        """
        set_cmd for the horizontal_scale
        """
        self.trace_ready = False
        self.write('HORizontal:SCAle {}'.format(scale))

    ##################################################
    # METHODS FOR THE USER                           #
    ##################################################

    def clear_message_queue(self, verbose=False):
        """
        Function to clear up (flush) the VISA message queue of the AWG
        instrument. Reads all messages in the the queue.

        Args:
            verbose (Bool): If True, the read messages are printed.
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
