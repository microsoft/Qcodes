import logging
import array as arr
import warnings

log = logging.getLogger(__name__)


from qcodes.instrument_drivers.tektronix.AWG5014 import Tektronix_AWG5014
from qcodes import validators as vals


class Tektronix_AWG5200(Tektronix_AWG5014):
    """
    This driver was a quick hack, but we now have another, more
    appropriate driver, namely the AWG5208 driver. Please refer to
    That one instead.

    """

    def __init__(self, name, address, timeout=180, num_channels=8, **kwargs):
        """
        Initializes the AWG5014.

        Args:
            name (string): name of the instrument
            address (string): GPIB or ethernet address as used by VISA
            timeout (float): visa timeout, in secs. long default (180)
                to accommodate large waveforms
            num_channels (int): number of channels on the device

        Returns:
            None
        """

        warnings.warn('This driver, Tektronix_AWG5200, is deprecated and will '
                      'be removed in the future. Please use the AWG5208 driver'
                      ' instead.')

        super().__init__(name, address, timeout=timeout, num_channels=num_channels, **kwargs)

        for i in range(1, self.num_channels + 1):
            resolution_cmd = 'SOURCE{}:DAC:RESOLUTION'.format(i)
            self.add_parameter('ch{}_resolution'.format(i),
                               label='Resultion for channel {}'.format(i),
                               get_cmd=resolution_cmd + '?',
                               set_cmd=resolution_cmd + ' {}',
                               vals=vals.Ints(0, 17),
                               get_parser=int)
            # this driver only supports 14-bit resolution (e.g. 2 marker
            # channels)
            self.set('ch{}_resolution'.format(i), 14)

    def send_waveform_to_list(self, w, m1, m2, wfmname):
        """
        Send a single complete waveform directly to the "User defined"
        waveform list (prepend it). The data type of the input arrays
        is unimportant, but the marker arrays must contain only 1's
        and 0's.

        The 5200 has support for upto 4 marker channels, but this is not
        supported at the moment.

        Args:
            w (numpy.ndarray): The waveform
            m1 (numpy.ndarray): Marker1
            m2 (numpy.ndarray): Marker2
            wfmname (str): waveform name

        Raises:
            Exception: if the lengths of w, m1, and m2 don't match
            TypeError: if the waveform contains values outside (-1, 1)
            TypeError: if the markers contain values that are not 0 or 1
        """
       # log.debug('Sending waveform {} to instrument'.format(wfmname))
        # Check for errors
        dim = len(w)

        # Input validation
        if (not((len(w) == len(m1)) and ((len(m1) == len(m2))))):
            raise Exception('error: sizes of the waveforms do not match')
        if min(w) < -1 or max(w) > 1:
            raise TypeError('Waveform values out of bonds.' +
                            ' Allowed values: -1 to 1 (inclusive)')
        if (list(m1).count(0) + list(m1).count(1)) != len(m1):
            raise TypeError('Marker 1 contains invalid values.' +
                            ' Only 0 and 1 are allowed')
        if (list(m2).count(0) + list(m2).count(1)) != len(m2):
            raise TypeError('Marker 2 contains invalid values.' +
                            ' Only 0 and 1 are allowed')

        self._values['files'][wfmname] = self._file_dict(w, m1, m2, None)

        # if we create a waveform with the same name but different size,
        # it will not get over written
        # Delete the possibly existing file (will do nothing if the file
        # doesn't exist
        s = 'WLISt:WAVeform:DEL "{}"'.format(wfmname)
        self.write(s)

        # create the waveform
        s = 'WLISt:WAVeform:NEW "{}",{:d},REAL'.format(
            wfmname, dim)  # was INT insted of real
        self.write(s)
        # Prepare the data block
        number = w
        number = number.astype('float32')  # was 'int' before
        ws = arr.array('f', number)  # was 'H' before

        ws = ws.tobytes()
        s1 = 'WLISt:WAVeform:DATA "{}",'.format(wfmname)
        s1 = s1.encode('UTF-8')
        s3 = ws
        s2 = '#' + str(len(str(len(s3)))) + str(len(s3))
        s2 = s2.encode('UTF-8')

        mes = s1 + s2 + s3
        self.visa_handle.write_raw(mes)

        number = m1 * 128 + m2 * 64  # valid for 2 marker configuration
        number = number.astype('uint8')
        ws = arr.array('B', number)

        ws = ws.tobytes()
        s1 = 'WLISt:WAVeform:MARKer:DATA "{}",'.format(wfmname)
        s1 = s1.encode('UTF-8')
        s3 = ws
        s2 = '#' + str(len(str(len(s3)))) + str(len(s3))
        s2 = s2.encode('UTF-8')

        mes = s1 + s2 + s3
        self.visa_handle.write_raw(mes)
