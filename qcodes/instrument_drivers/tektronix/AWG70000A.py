import xml.etree.ElementTree as ET
import datetime as dt
import numpy as np
import struct
import io
import zipfile as zf
import logging
from functools import partial

import time
from typing import List, Sequence

from qcodes import Instrument, VisaInstrument, validators as vals
from qcodes.instrument.channel import InstrumentChannel, ChannelList
from qcodes.utils.validators import Validator

log = logging.getLogger(__name__)

##################################################
#
# MODEL DEPENDENT SETTINGS
#

_fg_path_val_map = {'5208': {'DC High BW': "DCHB",
                             'DC High Voltage': "DCHV",
                             'AC Direct': "ACD"},
                    '70001A': {'direct': 'DIR',
                               'DCamplified': 'DCAM',
                               'AC': 'AC'},
                    '70002A': {'direct': 'DIR',
                               'DCamplified': 'DCAM',
                               'AC': 'AC'}}

# number of markers per channel
_num_of_markers_map = {'5208': 4,
                       '70001A': 2,
                       '70002A': 2}

# channel resolution
_chan_resolutions = {'5208': [12, 13, 14, 15, 16],
                     '70001A': [8, 9, 10],
                     '70002A': [8, 9, 10]}


class SRValidator(Validator):
    """
    Validator to validate the AWG clock sample rate
    """

    def __init__(self, awg: 'AWG70000A') -> None:
        """
        Args:
            awg: The parent instrument instance. We need this since sample
                rate validation depends on many clock settings
        """
        self.awg = awg
        if self.awg.model == '70001A':
            self._internal_validator = vals.Numbers(1.49e3, 50e9)
            self._freq_multiplier = 4
        elif self.awg.model == '70002A':
            self._internal_validator = vals.Numbers(1.49e3, 25e9)
            self._freq_multiplier = 2
        elif self.awg.model == '5208':
            self._internal_validator = vals.Numbers(1.49e3, 2.5e9)
        # no other models are possible, since the __init__ of
        # the AWG70000A raises an error if anything else is given

    def validate(self, value: float, context: str='') -> None:
        if 'Internal' in self.awg.clock_source():
            self._internal_validator.validate(value)
        else:
            ext_freq = self.awg.clock_external_frequency()
            # TODO: I'm not sure what the minimal allowed sample rate is
            # in this case
            validator = vals.Numbers(1.49e3, self._freq_multiplier*ext_freq)
            validator.validate(value)


class AWGChannel(InstrumentChannel):
    """
    Class to hold a channel of the AWG.
    """

    def __init__(self,  parent: Instrument, name: str, channel: int) -> None:
        """
        Args:
            parent: The Instrument instance to which the channel is
                to be attached.
            name: The name used in the DataSet
            channel: The channel number, either 1 or 2.
        """

        super().__init__(parent, name)

        self.channel = channel

        num_channels = self.root_instrument.num_channels
        self.model = self.root_instrument.model

        fg = 'function generator'

        if channel not in list(range(1, num_channels+1)):
            raise ValueError('Illegal channel value.')

        self.add_parameter('state',
                           label='Channel {} state'.format(channel),
                           get_cmd='OUTPut{}:STATe?'.format(channel),
                           set_cmd='OUTPut{}:STATe {{}}'.format(channel),
                           vals=vals.Ints(0, 1),
                           get_parser=int)

        ##################################################
        # FGEN PARAMETERS

        # TODO: Setting high and low will change this parameter's value
        self.add_parameter('fgen_amplitude',
                           label='Channel {} {} amplitude'.format(channel, fg),
                           get_cmd='FGEN:CHANnel{}:AMPLitude?'.format(channel),
                           set_cmd='FGEN:CHANnel{}:AMPLitude {{}}'.format(channel),
                           unit='V',
                           vals=vals.Numbers(0, 0.5),
                           get_parser=float)

        self.add_parameter('fgen_offset',
                           label='Channel {} {} offset'.format(channel, fg),
                           get_cmd='FGEN:CHANnel{}:OFFSet?'.format(channel),
                           set_cmd='FGEN:CHANnel{}:OFFSet {{}}'.format(channel),
                           unit='V',
                           vals=vals.Numbers(0, 0.250),  # depends on ampl.
                           get_parser=float)

        self.add_parameter('fgen_frequency',
                           label='Channel {} {} frequency'.format(channel, fg),
                           get_cmd='FGEN:CHANnel{}:FREQuency?'.format(channel),
                           set_cmd=partial(self._set_fgfreq, channel),
                           unit='Hz',
                           get_parser=float)

        self.add_parameter('fgen_dclevel',
                           label='Channel {} {} DC level'.format(channel, fg),
                           get_cmd='FGEN:CHANnel{}:DCLevel?'.format(channel),
                           set_cmd='FGEN:CHANnel{}:DCLevel {{}}'.format(channel),
                           unit='V',
                           vals=vals.Numbers(-0.25, 0.25),
                           get_parser=float)

        self.add_parameter('fgen_signalpath',
                           label='Channel {} {} signal path'.format(channel, fg),
                           set_cmd='FGEN:CHANnel{}:PATH {{}}'.format(channel),
                           get_cmd='FGEN:CHANnel{}:PATH?'.format(channel),
                           val_mapping=_fg_path_val_map[self.root_instrument.model])

        self.add_parameter('fgen_period',
                           label='Channel {} {} period'.format(channel, fg),
                           get_cmd='FGEN:CHANnel{}:PERiod?'.format(channel),
                           unit='s',
                           get_parser=float)

        self.add_parameter('fgen_phase',
                           label='Channel {} {} phase'.format(channel, fg),
                           get_cmd='FGEN:CHANnel{}:PHASe?'.format(channel),
                           set_cmd='FGEN:CHANnel{}:PHASe {{}}'.format(channel),
                           unit='degrees',
                           vals=vals.Numbers(-180, 180),
                           get_parser=float)

        self.add_parameter('fgen_symmetry',
                           label='Channel {} {} symmetry'.format(channel, fg),
                           set_cmd='FGEN:CHANnel{}:SYMMetry {{}}'.format(channel),
                           get_cmd='FGEN:CHANnel{}:SYMMetry?'.format(channel),
                           unit='%',
                           vals=vals.Numbers(0, 100),
                           get_parser=float)

        self.add_parameter('fgen_type',
                           label='Channel {} {} type'.format(channel, fg),
                           set_cmd='FGEN:CHANnel{}:TYPE {{}}'.format(channel),
                           get_cmd='FGEN:CHANnel{}:TYPE?'.format(channel),
                           val_mapping={'SINE': 'SINE',
                                        'SQUARE': 'SQU',
                                        'TRIANGLE': 'TRI',
                                        'NOISE': 'NOIS',
                                        'DC': 'DC',
                                        'GAUSSIAN': 'GAUSS',
                                        'EXPONENTIALRISE': 'EXPR',
                                        'EXPONENTIALDECAY': 'EXPD',
                                        'NONE': 'NONE'})

        ##################################################
        # AWG PARAMETERS

        # this command internally uses power in dBm
        # the manual claims that this command only works in AC mode
        # (OUTPut[n]:PATH is AC), but I've tested that it does what
        # one would expect in DIR mode.
        self.add_parameter(
            'awg_amplitude',
            label='Channel {} AWG peak-to-peak amplitude'.format(channel),
            set_cmd='SOURCe{}:VOLTage {{}}'.format(channel),
            get_cmd='SOURce{}:VOLTage?'.format(channel),
            unit='V',
            get_parser=float,
            vals=vals.Numbers(0.250, 0.500))

        # markers
        for mrk in range(1, _num_of_markers_map[self.model]+1):

            self.add_parameter(
                'marker{}_high'.format(mrk),
                label='Channel {} marker {} high level'.format(channel, mrk),
                set_cmd='SOURce{}:MARKer{}:VOLTage:HIGH {{}}'.format(channel,
                                                                     mrk),
                get_cmd='SOURce{}:MARKer{}:VOLTage:HIGH?'.format(channel, mrk),
                unit='V',
                vals=vals.Numbers(-1.4, 1.4),
                get_parser=float)

            self.add_parameter(
                'marker{}_low'.format(mrk),
                label='Channel {} marker {} low level'.format(channel, mrk),
                set_cmd='SOURce{}:MARKer{}:VOLTage:LOW {{}}'.format(channel,
                                                                    mrk),
                get_cmd='SOURce{}:MARKer{}:VOLTage:LOW?'.format(channel, mrk),
                unit='V',
                vals=vals.Numbers(-1.4, 1.4),
                get_parser=float)

            self.add_parameter(
                'marker{}_waitvalue'.format(mrk),
                label='Channel {} marker {} wait state'.format(channel, mrk),
                set_cmd='OUTPut{}:WVALue:MARKer{} {{}}'.format(channel, mrk),
                get_cmd='OUTPut{}:WVALue:MARKer{}?'.format(channel, mrk),
                vals=vals.Enum('FIRST', 'LOW', 'HIGH'))

        ##################################################
        # MISC.

        self.add_parameter('resolution',
                           label='Channel {} bit resolution'.format(channel),
                           get_cmd='SOURce{}:DAC:RESolution?'.format(channel),
                           set_cmd='SOURce{}:DAC:RESolution {{}}'.format(channel),
                           vals=vals.Enum(*_chan_resolutions[self.model]),
                           get_parser=int,
                           docstring=("""
                                      8 bit resolution allows for two
                                      markers, 9 bit resolution
                                      allows for one, and 10 bit
                                      does NOT allow for markers"""))

    def _set_fgfreq(self, channel: int, frequency: float) -> None:
        """
        Set the function generator frequency
        """
        functype = self.fgen_type.get()
        if functype in ['SINE', 'SQUARE']:
            max_freq = 12.5e9
        else:
            max_freq = 6.25e9

        # validate
        if frequency < 1 or frequency > max_freq:
            raise ValueError('Can not set channel {} frequency to {} Hz.'
                             ' Maximum frequency for function type {} is {} '
                             'Hz, minimum is 1 Hz'.format(channel, frequency,
                                                          functype, max_freq))
        else:
            self.root_instrument.write(f'FGEN:CHANnel{channel}:'
                                       f'FREQuency {frequency}')

    def setWaveform(self, name: str) -> None:
        """
        Select a waveform from the waveform list to output on this channel

        Args:
            name: The name of the waveform
        """
        if name not in self.root_instrument.waveformList:
            raise ValueError('No such waveform in the waveform list')

        self.root_instrument.write(f'SOURce{self.channel}:CASSet:WAVeform "{name}"')

    def setSequenceTrack(self, seqname: str, tracknr: int) -> None:
        """
        Assign a track from a sequence to this channel.

        Args:
            seqname: Name of the sequence in the sequence list
            tracknr: Which track to use (1 or 2)
        """

        self.root_instrument.write(f'SOURCE{self.channel}:'
                                   f'CASSet:SEQuence "{seqname}"'
                                   f', {tracknr}')


class AWG70000A(VisaInstrument):
    """
    The QCoDeS driver for Tektronix AWG70000A series AWG's.

    The drivers for AWG70001A and AWG70002A should be subclasses of this
    general class.
    """

    def __init__(self, name: str, address: str, num_channels: int,
                 timeout: float=10, **kwargs) -> None:
        """
        Args:
            name: The name used internally by QCoDeS in the DataSet
            address: The VISA resource name of the instrument
            timeout: The VISA timeout time (in seconds)
            num_channels: Number of channels on the AWG
        """

        self.num_channels = num_channels

        super().__init__(name, address, timeout=timeout, terminator='\n',
                         **kwargs)

        # The 'model' value begins with 'AWG'
        self.model = self.IDN()['model'][3:]

        if self.model not in ['70001A', '70002A', '5208']:
            raise ValueError('Unknown model type: {}. Are you using '
                             'the right driver for your instrument?'
                             ''.format(self.model))

        self.add_parameter('current_directory',
                           label='Current file system directory',
                           set_cmd='MMEMory:CDIRectory "{}"',
                           get_cmd='MMEMory:CDIRectory?',
                           vals=vals.Strings())

        self.add_parameter('mode',
                           label='Instrument operation mode',
                           set_cmd='INSTrument:MODE {}',
                           get_cmd='INSTrument:MODE?',
                           vals=vals.Enum('AWG', 'FGEN'))

        ##################################################
        # Clock parameters

        self.add_parameter('sample_rate',
                           label='Clock sample rate',
                           set_cmd='CLOCk:SRATe {}',
                           get_cmd='CLOCk:SRATe?',
                           unit='Sa/s',
                           get_parser=float,
                           vals=SRValidator(self))

        self.add_parameter('clock_source',
                           label='Clock source',
                           set_cmd='CLOCk:SOURce {}',
                           get_cmd='CLOCk:SOURce?',
                           val_mapping={'Internal': 'INT',
                                        'Internal, 10 MHZ ref.': 'EFIX',
                                        'Internal, variable ref.': 'EVAR',
                                        'External': 'EXT'})

        self.add_parameter('clock_external_frequency',
                           label='External clock frequency',
                           set_cmd='CLOCk:ECLock:FREQuency {}',
                           get_cmd='CLOCk:ECLock:FREQuency?',
                           get_parser=float,
                           unit='Hz',
                           vals=vals.Numbers(6.25e9, 12.5e9))

        # We deem 2 channels too few for a channel list
        if self.num_channels > 2:
            chanlist = ChannelList(self, 'Channels', AWGChannel,
                                   snapshotable=False)

        for ch_num in range(1, num_channels+1):
            ch_name = 'ch{}'.format(ch_num)
            channel = AWGChannel(self, ch_name, ch_num)
            self.add_submodule(ch_name, channel)
            if self.num_channels > 2:
                chanlist.append(channel)

        if self.num_channels > 2:
            chanlist.lock()
            self.add_submodule('channels', chanlist)

        # Folder on the AWG where to files are uplaoded by default
        self.wfmxFileFolder = "\\Users\\OEM\\Documents"
        self.seqxFileFolder = "\\Users\\OEM\Documents"

        self.current_directory(self.wfmxFileFolder)

        self.connect_message()

    def force_triggerA(self):
        """
        Force a trigger A event
        """
        self.write('TRIGger:IMMediate ATRigger')

    def force_triggerB(self):
        """
        Force a trigger B event
        """
        self.write('TRIGger:IMMediate BTRigger')

    def play(self) -> None:
        """
        Run the AWG/Func. Gen. This command is equivalent to pressing the
        play button on the front panel.
        """
        self.write('AWGControl:RUN')

    def stop(self) -> None:
        """
        Stop the output of the instrument. This command is equivalent to
        pressing the stop button on the front panel.
        """
        self.write('AWGControl:STOP')

    @property
    def sequenceList(self) -> List[str]:
        """
        Return the sequence list as a list of strings
        """
        # There is no SLISt:LIST command, so we do it slightly differently
        N = int(self.ask("SLISt:SIZE?"))
        slist = []
        for n in range(1, N+1):
            resp = self.ask("SLISt:NAME? {}".format(n))
            resp = resp.strip()
            resp = resp.replace('"', '')
            slist.append(resp)

        return slist

    @property
    def waveformList(self) -> List[str]:
        """
        Return the waveform list as a list of strings
        """
        respstr = self.ask("WLISt:LIST?")
        respstr = respstr.strip()
        respstr = respstr.replace('"', '')
        resp = respstr.split(',')

        return resp

    def clearSequenceList(self):
        """
        Clear the sequence list
        """
        self.write('SLISt:SEQuence:DELete ALL')

    def clearWaveformList(self):
        """
        Clear the waveform list
        """
        self.write('WLISt:WAVeform:DELete ALL')

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
        elif len(shape) == 2:
            N = shape[1]
            markers_included = True
        else:
            raise ValueError('Input data has too many dimensions!')

        wfmx_hdr_str = AWG70000A._makeWFMXFileHeader(num_samples=N,
                                                     markers_included=markers_included)
        wfmx_hdr = bytes(wfmx_hdr_str, 'ascii')
        wfmx_data = AWG70000A._makeWFMXFileBinaryData(data, amplitude)

        wfmx = wfmx_hdr

        wfmx += wfmx_data

        return wfmx

    def sendSEQXFile(self, seqx: bytes, filename: str,
                     path: str=None) -> None:
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

    def sendWFMXFile(self, wfmx: bytes, filename: str,
                     path: str=None) -> None:
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

    def _sendBinaryFile(self, binfile: bytes, filename: str,
                        path: str) -> None:
        """
        Send a binary file to the AWG's mass memory (disk).

        Args:
            binfile: The binary file to send.
            filename: The name of the file on the AWG disk, including the
                extension.
            path: The path to the directory where the file should be saved.
        """

        name_str = 'MMEMory:DATA "{}"'.format(filename).encode('ascii')
        len_file = len(binfile)
        len_str = len(str(len_file))  # No. of digits needed to write length
        size_str = (',#{}{}'.format(len_str, len_file)).encode('ascii')

        msg = name_str + size_str + binfile

        # IEEE 488.2 limit on a single write is 999,999,999 bytes
        # TODO: If this happens, we should split the file
        if len(msg) > 1e9-1:
            raise ValueError('File too large to transfer')

        self.current_directory(path)

        self.visa_handle.write_raw(msg)

    def loadWFMXFile(self, filename: str, path: str=None) -> None:
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

        pathstr = 'C:' + path + '\\' + filename

        self.write('MMEMory:OPEN "{}"'.format(pathstr))
        # the above command is overlapping, but we want a blocking command
        self.ask("*OPC?")

    def loadSEQXFile(self, filename: str, path: str=None) -> None:
        """
        Load a seqx file from instrument disk memory. All sequences in the file
        are loaded into the sequence list.

        Args:
            filename: The name of the sequence file
            path: Path to load from. If omitted, the default path
                (self.seqxFileFolder) is used.
        """
        if not path:
            path = self.seqxFileFolder

        pathstr = 'C:{}\\{}'.format(path, filename)

        self.write('MMEMory:OPEN:SASSet:SEQuence "{}"'.format(pathstr))
        # the above command is overlapping, but we want a blocking command
        self.ask('*OPC?')

    @staticmethod
    def _makeWFMXFileHeader(num_samples: int,
                            markers_included: bool) -> str:
        """
        Compiles a valid XML header for a .wfmx file
        There might be behaviour we can't capture

        We always use 9 digits for the number of header character
        """
        offsetdigits = 9

        if not isinstance(num_samples, int):
            raise ValueError('num_samples must be of type int.')

        if num_samples < 2400:
            raise ValueError('num_samples must be at least 2400.')

        # form the timestamp string
        timezone = time.timezone
        tz_m, _ = divmod(timezone, 60)  # returns (minutes, seconds)
        tz_h, tz_m = divmod(tz_m, 60)
        if np.sign(tz_h) == -1:
            signstr = '-'
            tz_h *= -1
        else:
            signstr = '+'
        timestr = dt.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
        timestr += signstr
        timestr += '{:02.0f}:{:02.0f}'.format(tz_h, tz_m)

        hdr = ET.Element('DataFile', attrib={'offset': '0'*offsetdigits,
                                             'version': '0.1'})
        dsc = ET.SubElement(hdr, 'DataSetsCollection')
        dsc.set("xmlns", "http://www.tektronix.com")
        dsc.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        dsc.set("xsi:schemaLocation", (r"http://www.tektronix.com file:///" +
                                       r"C:\Program%20Files\Tektronix\AWG70000" +
                                       r"\AWG\Schemas\awgDataSets.xsd"))
        datasets = ET.SubElement(dsc, 'DataSets')
        datasets.set('version', '1')
        datasets.set("xmlns", "http://www.tektronix.com")

        # Description of the data
        datadesc = ET.SubElement(datasets, 'DataDescription')
        temp_elem = ET.SubElement(datadesc, 'NumberSamples')
        temp_elem.text = '{:d}'.format(num_samples)
        temp_elem = ET.SubElement(datadesc, 'SamplesType')
        temp_elem.text = 'AWGWaveformSample'
        temp_elem = ET.SubElement(datadesc, 'MarkersIncluded')
        temp_elem.text = ('{}'.format(markers_included)).lower()
        temp_elem = ET.SubElement(datadesc, 'NumberFormat')
        temp_elem.text = 'Single'
        temp_elem = ET.SubElement(datadesc, 'Endian')
        temp_elem.text = 'Little'
        temp_elem = ET.SubElement(datadesc, 'Timestamp')
        temp_elem.text = timestr

        # Product specific information
        prodspec = ET.SubElement(datasets, 'ProductSpecific')
        prodspec.set('name', '')
        temp_elem = ET.SubElement(prodspec, 'ReccSamplingRate')
        temp_elem.set('units', 'Hz')
        temp_elem.text = 'NaN'
        temp_elem = ET.SubElement(prodspec, 'ReccAmplitude')
        temp_elem.set('units', 'Volts')
        temp_elem.text = 'NaN'
        temp_elem = ET.SubElement(prodspec, 'ReccOffset')
        temp_elem.set('units', 'Volts')
        temp_elem.text = 'NaN'
        temp_elem = ET.SubElement(prodspec, 'SerialNumber')
        temp_elem = ET.SubElement(prodspec, 'SoftwareVersion')
        temp_elem.text = '1.0.0917'
        temp_elem = ET.SubElement(prodspec, 'UserNotes')
        temp_elem = ET.SubElement(prodspec, 'OriginalBitDepth')
        temp_elem.text = 'Floating'
        temp_elem = ET.SubElement(prodspec, 'Thumbnail')
        temp_elem = ET.SubElement(prodspec, 'CreatorProperties',
                          attrib={'name': ''})
        temp_elem = ET.SubElement(hdr, 'Setup')

        xmlstr = ET.tostring(hdr, encoding='unicode')
        xmlstr = xmlstr.replace('><', '>\r\n<')

        # As the final step, count the length of the header and write this
        # in the DataFile tag attribute 'offset'

        xmlstr = xmlstr.replace('0'*offsetdigits,
                                '{num:0{pad}d}'.format(num=len(xmlstr),
                                                       pad=offsetdigits))

        return xmlstr

    @staticmethod
    def _makeWFMXFileBinaryData(data: np.ndarray, amplitude: float) -> bytes:
        """
        For the binary part.

        Note that currently only zero markers or two markers are supported;
        one-marker data will break.

        Args:
            data: Either a shape (N,) array with only a waveform or
                a shape (3, N) array with waveform, marker1, marker2, i.e.
                data = np.array([wfm, m1, m2]). The waveform data is assumed
                to be in V.
            amplitude: The peak-to-peak amplitude (V) assumed to be set on the
                channel that will play this waveform. This information is
                needed as the waveform must be rescaled to (-1, 1) where
                -1 will correspond to the channel's min. voltage and 1 to the
                channel's max. voltage.
        """

        channel_max = amplitude/2
        channel_min = -amplitude/2

        shape = np.shape(data)

        if len(shape) == 1:
            N = shape[0]
            binary_marker = b''
            wfm = data
        else:
            N = shape[1]
            M = shape[0]
            wfm = data[0, :]
            if M == 2:
                markers = data[1, :]
            elif M == 3:
                m1 = data[1, :]
                m2 = data[2, :]
                markers = m1+2*m2  # This is how one byte encodes both markers
                markers = markers.astype(int)
            fmt = N*'B'  # endian-ness doesn't matter for one byte
            binary_marker = struct.pack(fmt, *markers)

        if wfm.max() > channel_max or wfm.min() < channel_min:
            log.warning('Waveform exceeds specified channel range.'
                        ' The resulting waveform will be clipped. '
                        'Waveform min.: {} (V), waveform max.: {} (V),'
                        'Channel min.: {} (V), channel max.: {} (V)'
                        ''.format(wfm.min(), wfm.max(), channel_min,
                                  channel_max))

        # the data must be such that channel_max becomes 1 and
        # channel_min becomes -1
        scale = 2/amplitude
        wfm = wfm*scale

        # TODO: Is this a fast method?
        fmt = '<' + N*'f'
        binary_wfm = struct.pack(fmt, *wfm)
        binary_out = binary_wfm + binary_marker

        return binary_out

    @staticmethod
    def makeSEQXFile(trig_waits: Sequence[int],
                     nreps: Sequence[int],
                     event_jumps: Sequence[int],
                     event_jump_to: Sequence[int],
                     go_to: Sequence[int],
                     wfms: Sequence[Sequence[np.ndarray]],
                     amplitudes: Sequence[float],
                     seqname: str) -> bytes:
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

        Returns:
            The binary .seqx file, ready to be sent to the instrument.
        """

        # input sanitising to avoid spaces in filenames
        seqname = seqname.replace(' ', '_')

        # np.shape(wfms) returns
        # (no_of_chans, no_of_elms, no_of_arrays, no_of_points)
        # where no_of_arrays is 3 if both markers are included
        (chans, elms) = np.shape(wfms)[0: 2]
        wfm_names = [['wfmch{}pos{}'.format(ch, el) for el in range(1, elms+1)]
                     for ch in range(1, chans+1)]

        # generate wfmx files for the waveforms
        flat_wfmxs = [] # type: List[bytes]
        for amplitude, wfm_lst in zip(amplitudes, wfms):
            flat_wfmxs += [AWG70000A.makeWFMXFile(wfm, amplitude)
                           for wfm in wfm_lst]

        flat_wfm_names = [name for lst in wfm_names for name in lst]

        sml_file = AWG70000A._makeSMLFile(trig_waits, nreps,
                                          event_jumps, event_jump_to,
                                          go_to, wfm_names, seqname)

        user_file = b''
        setup_file = AWG70000A._makeSetupFile(seqname)

        buffer = io.BytesIO()

        zipfile = zf.ZipFile(buffer, mode='a')
        zipfile.writestr('Sequences/{}.sml'.format(seqname), sml_file)

        for (name, wfile) in zip(flat_wfm_names, flat_wfmxs):
            zipfile.writestr('Waveforms/{}.wfmx'.format(name), wfile)

        zipfile.writestr('setup.xml', setup_file)
        zipfile.writestr('userNotes.txt', user_file)
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
        head = ET.Element('RSAPersist')
        head.set('version', '0.1')
        temp_elem = ET.SubElement(head, 'Application')
        temp_elem.text = 'Pascal'
        temp_elem = ET.SubElement(head, 'MainSequence')
        temp_elem.text = sequence
        prodspec = ET.SubElement(head, 'ProductSpecific')
        prodspec.set('name', 'AWG70002A')
        temp_elem = ET.SubElement(prodspec, 'SerialNumber')
        temp_elem.text = 'B020397'
        temp_elem = ET.SubElement(prodspec, 'SoftwareVersion')
        temp_elem.text = '5.3.0128.0'
        temp_elem = ET.SubElement(prodspec, 'CreatorProperties')
        temp_elem.set('name', '')

        xmlstr = ET.tostring(head, encoding='unicode')
        xmlstr = xmlstr.replace('><', '>\r\n<')

        return xmlstr

    @staticmethod
    def _makeSMLFile(trig_waits: Sequence[int],
                     nreps: Sequence[int],
                     event_jumps: Sequence[int],
                     event_jump_to: Sequence[int],
                     go_to: Sequence[int],
                     wfm_names: Sequence[Sequence[str]],
                     seqname: str) -> str:
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
            wfm_names: The waveforms to use. Should be packed like
                [[wfmch1pos1, wfmch1pos2, ...], [wfmch2pos1, ...], ...]
            seqname: The name of the sequence. This name will appear in
                the sequence list of the instrument.

        Returns:
            A str containing the file contents, to be saved as an .sml file
        """
        offsetdigits = 9

        waitinputs = {0: 'None', 1: 'TrigA', 2: 'TrigB', 3: 'Internal'}
        eventinputs = {0: 'None', 1: 'TrigA', 2: 'TrigB', 3: 'Internal'}

        inputlsts = [trig_waits, nreps, event_jump_to, go_to]
        lstlens = [len(lst) for lst in inputlsts]
        if lstlens.count(lstlens[0]) != len(lstlens):
            raise ValueError('All input lists must have the same length!')

        if lstlens[0] == 0:
            raise ValueError('Received empty sequence option lengths!')

        # hackish check of wmfs dimensions
        if len(np.shape(wfm_names)) != 2:
            raise ValueError('Wrong shape of wfm_names input argument.')

        if lstlens[0] != np.shape(wfm_names)[1]:
            raise ValueError('Mismatch between number of waveforms and'
                             ' number of sequencing steps.')

        N = lstlens[0]
        chans = np.shape(wfm_names)[0]

        # for easy indexing later
        wfm_names_arr = np.array(wfm_names)

        # form the timestamp string
        timezone = time.timezone
        tz_m, _ = divmod(timezone, 60)
        tz_h, tz_m = divmod(tz_m, 60)
        if np.sign(tz_h) == -1:
            signstr = '-'
            tz_h *= -1
        else:
            signstr = '+'
        timestr = dt.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
        timestr += signstr
        timestr += '{:02.0f}:{:02.0f}'.format(tz_h, tz_m)

        datafile = ET.Element('DataFile', attrib={'offset': '0'*offsetdigits,
                                                  'version': '0.1'})
        dsc = ET.SubElement(datafile, 'DataSetsCollection')
        dsc.set("xmlns", "http://www.tektronix.com")
        dsc.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        dsc.set("xsi:schemaLocation", (r"http://www.tektronix.com file:///" +
                                       r"C:\Program%20Files\Tektronix\AWG70000" +
                                       r"\AWG\Schemas\awgSeqDataSets.xsd"))
        datasets = ET.SubElement(dsc, 'DataSets')
        datasets.set('version', '1')
        datasets.set("xmlns", "http://www.tektronix.com")

        # Description of the data
        datadesc = ET.SubElement(datasets, 'DataDescription')
        temp_elem = ET.SubElement(datadesc, 'SequenceName')
        temp_elem.text = seqname
        temp_elem = ET.SubElement(datadesc, 'Timestamp')
        temp_elem.text = timestr
        temp_elem = ET.SubElement(datadesc, 'JumpTiming')
        temp_elem.text = 'JumpImmed'  # TODO: What does this control?
        temp_elem = ET.SubElement(datadesc, 'RecSampleRate')
        temp_elem.text = 'NaN'
        temp_elem = ET.SubElement(datadesc, 'RepeatFlag')
        temp_elem.text = 'false'
        temp_elem = ET.SubElement(datadesc, 'PatternJumpTable')
        temp_elem.set('Enabled', 'false')
        temp_elem.set('Count', '65536')
        steps = ET.SubElement(datadesc, 'Steps')
        steps.set('StepCount', '{:d}'.format(N))
        steps.set('TrackCount', '{:d}'.format(chans))

        for n in range(1, N+1):
            step = ET.SubElement(steps, 'Step')
            temp_elem = ET.SubElement(step, 'StepNumber')
            temp_elem.text = '{:d}'.format(n)
            # repetitions
            rep = ET.SubElement(step, 'Repeat')
            repcount = ET.SubElement(step, 'RepeatCount')
            if nreps[n-1] == 0:
                rep.text = 'Infinite'
                repcount.text = '1'
            elif nreps[n-1] == 1:
                rep.text = 'Once'
                repcount.text = '1'
            else:
                rep.text = 'RepeatCount'
                repcount.text = '{:d}'.format(nreps[n-1])
            # trigger wait
            temp_elem = ET.SubElement(step, 'WaitInput')
            temp_elem.text = waitinputs[trig_waits[n-1]]
            # event jump
            temp_elem = ET.SubElement(step, 'EventJumpInput')
            temp_elem.text = eventinputs[event_jumps[n-1]]
            jumpto = ET.SubElement(step, 'EventJumpTo')
            jumpstep = ET.SubElement(step, 'EventJumpToStep')
            if event_jump_to[n-1] == 0:
                jumpto.text = 'Next'
                jumpstep.text = '1'
            else:
                jumpto.text = 'StepIndex'
                jumpstep.text = '{:d}'.format(event_jump_to[n-1])
            # Go to
            goto = ET.SubElement(step, 'GoTo')
            gotostep = ET.SubElement(step, 'GoToStep')
            if go_to[n-1] == 0:
                goto.text = 'Next'
                gotostep.text = '1'
            else:
                goto.text = 'StepIndex'
                gotostep.text = '{:d}'.format(go_to[n-1])

            assets = ET.SubElement(step, 'Assets')
            for wfm in wfm_names_arr[:, n-1]:
                asset = ET.SubElement(assets, 'Asset')
                temp_elem = ET.SubElement(asset, 'AssetName')
                temp_elem.text = wfm
                temp_elem = ET.SubElement(asset, 'AssetType')
                temp_elem.text = 'Waveform'

            flags = ET.SubElement(step, 'Flags')
            for _ in range(chans):
                flagset = ET.SubElement(flags, 'FlagSet')
                for flg in ['A', 'B', 'C', 'D']:
                    temp_elem = ET.SubElement(flagset, 'Flag')
                    temp_elem.set('name', flg)
                    temp_elem.text = 'NoChange'

        temp_elem = ET.SubElement(datasets, 'ProductSpecific')
        temp_elem.set('name', '')
        temp_elem = ET.SubElement(datafile, 'Setup')

        xmlstr = ET.tostring(datafile, encoding='unicode')
        xmlstr = xmlstr.replace('><', '>\r\n<')

        # As the final step, count the length of the header and write this
        # in the DataFile tag attribute 'offset'

        xmlstr = xmlstr.replace('0'*offsetdigits,
                                '{num:0{pad}d}'.format(num=len(xmlstr),
                                                       pad=offsetdigits))

        return xmlstr
