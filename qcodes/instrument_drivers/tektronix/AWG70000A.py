import xml.etree.ElementTree as ET
import datetime as dt
import numpy as np

from dateutil.tz import time

from qcodes import Instrument, VisaInstrument, validators as vals
from qcodes.instrument.channel import ChannelList, InstrumentChannel


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

        num_channels = self._parent.num_channels

        if channel not in list(range(1, num_channels+1)):
            raise ValueError('Illegal channel value.')

        self.add_parameter('state',
                           label='Channel {} state'.format(channel),
                           get_cmd='OUTPut{}:STATe?'.format(channel),
                           set_cmd='OUTPut{}:STATe {{}}'.format(channel),
                           vals=vals.Ints(0, 1),
                           get_parser=int)

        # TODO: Setting high and low will change this parameter's value
        self.add_parameter('amplitude',
                           label='Channel {} amplitude'.format(channel),
                           get_cmd='FGEN:CHANnel{}:AMPLitude?'.format(channel),
                           set_cmd='FGEN:CHANnel{}:AMPLitude {{}}'.format(channel),
                           vals=vals.Numbers(0, 0.5),
                           get_parser=float)

        self.add_parameter('frequency',
                           label='Channel {} frequency'.format(channel),
                           get_cmd='FGEN:CHANnel{}:FREQuency?'.format(channel),
                           set_cmd='FGEN:CHANnel{}:FREQuency {{}}'.format(channel),
                           vals=vals.Numbers(1, 50000000),
                           get_parser=float)

        self.add_parameter('dclevel',
                           label='Channel {} DC level'.format(channel),
                           get_cmd='FGEN:CHANnel{}:DCLevel?'.format(channel),
                           set_cmd='FGEN:CHANnel{}:DCLevel {{}}'.format(channel),
                           vals=vals.Numbers(-0.25, 0.25),
                           get_parser=float)


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

        super().__init__(name, address, timeout=timeout, **kwargs)

        for ch_num in range(1, num_channels+1):
            ch_name = 'ch{}'.format(ch_num)
            channel = AWGChannel(self, ch_name, ch_num)
            self.add_submodule(ch_name, channel)

        self.connect_message()

    # Idea: two low-level parsers, one for the header and one for the data.

    def makeWFXFile(self, data: np.ndarray) -> bytes:
        """
        Compose a WFMX file

        Args:
            data: A numpy array holding the data. Markers can be included.
        """

        N = np.shape(data)[1]
        layers = np.shape(data)[0]

        if layers > 1:
            markers_included = True

        wfmx_hdr = self._makeWFMXFileHeader(num_samples=N,
                                            markers_included=markers_included)
        wfmx_hdr = bytes(wfmx_hdr, 'utf8')
        wfmx_data = self._makeWFMXFileBinaryData(data)

        wfmx = wfmx_hdr + wfmx_data

        return wfmx

    def _makeWFMXFileHeader(self,
                            num_samples: int,
                            markers_included: bool) -> str:
        """
        beta version

        Try to compile a valid XML header for a .wfmx file

        We always use 9 digits for the number of header character
        """
        offsetdigits = 9

        if not isinstance(num_samples, int):
            raise ValueError('num_samples must be of type int.')

        # form the timestamp string
        timezone = time.timezone
        tz_m, tz_s = divmod(timezone, 60)
        tz_h, tz_m = divmod(tz_m, 60)
        if np.sign(tz_h) == -1:
            signstr = '-'
            tz_h *= -1
        else:
            signstr = '+'
        timestr = dt.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%s')
        timestr += signstr
        timestr += '{:02.0f}:{:02.0f}'.format(tz_h, tz_m)

        hdr = ET.Element('DataFile', attrib={'offset': '0'*offsetdigits,
                                             'version': '0.2'})
        _ = ET.SubElement(hdr, 'DataSetsCollection')
        datasets = ET.SubElement(_, 'DataSets', attrib={'version': '1'})

        # Description of the data
        datadesc = ET.SubElement(datagsets, 'DataDescription')
        _ = ET.SubElement(datadesc, 'NumberSamples')
        _.text = '{:d}'.format(num_samples)
        _ = ET.SubElement(datadesc, 'SamplesType')
        _.text = 'AWGWaveformSample'
        _ = ET.SubElement(datadesc, 'MarkersIncluded')
        _.text = ('{}'.format(markers_included)).lower()
        _ = ET.SubElement(datadesc, 'NumberFormat')
        _.text = 'Single'
        _ = ET.SubElement(datadesc, 'Endian')
        _.text = 'Little'
        _ = ET.SubElement(datadesc, 'Timestamp')
        _.text = timestr

        # Product specific information
        prodspec = ET.SubElement(datasets, 'ProductSpecific')
        _ = ET.SubElement(prodspec, 'CreatorProperties',
                          attrib={'name': 'QCoDeS'})

        xmlstr = ET.tostringlist(hdr)[0].decode('utf8')
        xmlstr = xmlstr.replace('><', '>\n<')  # TODO: Make this \r\n?

        # As the final step, count the length of the header and write this
        # in the DataFile tag attribute 'offset'

        xmlstr = xmlstr.replace('0'*offsetdigits,
                                '{num:0{pad}d}'.format(num=len(xmlstr),
                                                       pad=offsetdigits))

        return xmlstr

    def _makeWFMXFileBinaryData(self) -> bytes:
        """
        For the binary part
        """
        pass
