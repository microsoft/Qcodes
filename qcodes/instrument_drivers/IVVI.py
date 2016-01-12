import time
import numpy as np
import visa  # used for the parity constant
# load the qcodes path, until we have this installed as a package
import sys
qcpath = 'D:\GitHubRepos\Qcodes'
if qcpath not in sys.path:
    sys.path.append(qcpath)

from qcodes.instrument.visa import VisaInstrument
from qcodes.utils import validators as vals


class IVVI(VisaInstrument):
    '''
    Status: Alpha version, tested for basic get-set commands
        TODO:
            - Add individual parameters per channel
            - Add individual parameters for channel polarities
            - Add range protection per channel (adjustable for next version)
            - Add ramping speed protection (mV/s parameter for each channel)
            - Add error handling for the specific error messages in the protocol
            - Remove/fine-tune manual sleep statements

    This is the python driver for the D5 module of the IVVI-rack
    see: http://qtwork.tudelft.nl/~schouten/ivvi/doc-d5/index-d5.htm

    A descriptor for the data protocol can be found at
    http://qtwork.tudelft.nl/~schouten/ivvi/doc-d5/rs232linkformat.txt
    '''

    def __init__(self, name, address, reset=False, numdacs=8,
                 polarity=['BIP', 'BIP', 'BIP', 'BIP']):
        '''
        Initialzes the IVVI, and communicates with the wrapper
        Input:
            name (string)        : name of the instrument
            address (string)     : ASRL address
            reset (bool)         : resets to default values, default=false
            numdacs (int)        : number of dacs, multiple of 4, default=8
            polarity (string[4]) : list of polarities of each set of 4 dacs
                                   choose from 'BIP', 'POS', 'NEG',
                                   default=['BIP', 'BIP', 'BIP', 'BIP']
        '''
        super().__init__(name, address)
        # Set parameters
        self._address = address
        if numdacs % 4 == 0 and numdacs > 0:
            self._numdacs = int(numdacs)
        self.pol_num = np.zeros(self._numdacs)  # corresponds to POS polarity
        self.set_pol_dacrack('BIP', range(numdacs))

        # self.visa_handle= rm.open_resource('ASRL1')
        self.visa_handle.baud_rate = 115200
        self.visa_handle.parity = visa.constants.Parity(1)  # odd parity
        # Add parameters
        self.add_parameter('version',
                           get_cmd=self._get_version)
        self.add_parameter('dac voltages',
                           label='Dac voltages (mV)',
                           get_cmd=self._get_dacs)
        # for i in range(numdacs):
        #     self.add_parameter('ch{}'.format(i+1),
        #                        label='Dac {} (mV)'.format(i+1),
        #                        get_cmd=self.get_dac)
        #                        set_cmd=self.set_dac
        #                        parse_function=

    def __del__(self):
        '''
        Closes up the IVVI driver

        Input:
            None

        Output:
            None
        '''
        self.visa_handle.close()

    def _get_version(self):
        mes = self.ask(bytes([3, 4]))
        v = mes[2]
        print(v)
        return v

    def get_all(self):
        for par in self.parameters:
                        self[par].get()
        return self.snapshot()

    def set_dacs_zero(self):
        for i in range(self._numdacs):
            self.set_dac(i+1, 0)

    # Conversion of data
    def _mvoltage_to_bytes(self, mvoltage):
        '''
        Converts a mvoltage on a 0mV-4000mV scale to a 16-bit integer equivalent
        output is a list of two bytes

        Input:
            mvoltage (float) : a mvoltage in the 0mV-4000mV range

        Output:
            (dataH, dataL) (int, int) : The high and low value byte equivalent
        '''
        bytevalue = int(round(mvoltage/4000.0*65535))
        return bytevalue.to_bytes(length=2, byteorder='big')

    def _bytes_to_mvoltages(self, numbers):
        '''
        Converts a list of bytes to a list containing
        the corresponding mvoltages
        '''
        values = list(range(self._numdacs))
        for i in range(self._numdacs):
            values[i] = ((numbers[2 + 2*i]*256 + numbers[3 + 2*i]) /
                         65535.0*4000.0) + self.pol_num[i]
        return values

    # Communication with device
    def get_dac(self, channel):
        '''
        Returns dac channel in mV
        channels range from 1-numdacs

        TODO add a soft version  that only looks at the values in memory instead
        of getting all values in order to return one.
        '''
        dac_val = self._get_dacs[channel-1]
        return dac_val

    def set_dac(self, channel, mvoltage):
        '''
        Sets the specified dac to the specified voltage

        Input:
            mvoltage (float) : output voltage in mV
            channel (int)    : 1 based index of the dac

        Output:
            reply (string) : errormessage
        Private version of function
        '''
        byte_val = self._mvoltage_to_bytes(mvoltage - self.pol_num[channel-1])
        message = bytes([2, 1, channel]) + byte_val
        time.sleep(.2)
        reply = self.ask(message)
        return reply

    def _get_dacs(self):
        '''
        Reads from device and returns all dacvoltages in a list

        Input:
            None

        Output:
            voltages (float[]) : list containing all dacvoltages (in mV)

        get dacs command takes ~450ms according to ipython timeit
        '''
        message = bytes([self._numdacs*2+2, 2])
        reply = self.ask(message)
        # return reply
        mvoltages = self._bytes_to_mvoltages(reply)
        return mvoltages

    def write(self, raw_message, raw=False):
        '''
        Protocol specifies that a write consists of
        descriptor size, error_code, message

        returns message_len
        '''
        message_len = len(raw_message)+2
        error_code = bytes([0])
        message = bytes([message_len]) + error_code + raw_message
        if raw:
             self.visa_handle.write_raw(raw_message)
        else:
            self.visa_handle.write_raw(message)

        answer_length = raw_message[0]
        return answer_length

    def ask(self, message, raw=False):
        '''
        Send <message> to the device and read answer.
        Raises an error if one occurred
        Returns a list of bytes
        '''
        message_len = self.write(message, raw=raw)
        return self.read(message_len)

    def read(self, message_len=None):
        # because protocol has no termination chars the read reads the number
        # of bytes in the buffer
        time.sleep(.4)
        bytes_in_buffer = 0
        timeout = 1
        t0 = time.time()
        t1 = t0
        i=0
        while bytes_in_buffer == 0:
            t1 = time.time()
            time.sleep(.05)
            bytes_in_buffer = self.visa_handle.bytes_in_buffer
            if t1-t0 > timeout:
                raise TimeoutError()
        if message_len is None:
            message_len = bytes_in_buffer
        # a workaround for a timeout error in the pyvsia read_raw() function
        with(self.visa_handle.ignore_warning(visa.constants.VI_SUCCESS_MAX_CNT)):
            mes = self.visa_handle.visalib.read(
                self.visa_handle.session, message_len)
        mes = mes[0]  # cannot be done on same line for some reason
        # if mes[1] != 0:
        #     # see protocol descriptor for error codes
        #     raise Exception('IVVI rack exception "%s"' % mes[1])
        return mes

    def set_pol_dacrack(self, flag, channels, getall=True):
        '''
        Changes the polarity of the specified set of dacs

        Input:
            flag (string) : 'BIP', 'POS' or 'NEG'
            channel (int) : 0 based index of the rack
            getall (boolean): if True (default) perform a get_all

        Output:
            None
        '''
        flagmap = {'NEG': -4000, 'BIP': -2000, 'POS': 0}
        if flag.upper() not in flagmap:
            raise KeyError('Tried to set invalid dac polarity %s', flag)

        val = flagmap[flag.upper()]
        for ch in channels:
            self.pol_num[ch-1] = val
            # self.set_parameter_bounds('dac%d' % (i+1), val, val + 4000.0)

        if getall:
            self.get_all()

    def get_pol_dac(self, channel):
        '''
        Returns the polarity of the dac channel specified

        Input:
            channel (int) : 1 based index of the dac

        Output:
            polarity (string) : 'BIP', 'POS' or 'NEG'
        '''
        val = self.pol_num[channel-1]

        if (val == -4000):
            return 'NEG'
        elif (val == -2000):
            return 'BIP'
        elif (val == 0):
            return 'POS'
        else:
            return 'Invalid polarity in memory'

    # def byte_limited_arange(self, start, stop, step=1, pol=None, dacnr=None):
    #     '''
    #     Creates array of mvoltages, in integer steps of the dac resolution. Either
    #     the dac polarity, or the dacnr needs to be specified.
    #     '''
    #     if pol is not None and dacnr is not None:
    #         logging.error('byte_limited_arange: speficy "pol" OR "dacnr", NOT both!')
    #     elif pol is None and dacnr is None:
    #         logging.error('byte_limited_arange: need to specify "pol" or "dacnr"')
    #     elif dacnr is not None:
    #         pol = self.get_pol_dac(dacnr)

    #     if (pol.upper() == 'NEG'):
    #         polnum = -4000
    #     elif (pol.upper() == 'BIP'):
    #         polnum = -2000
    #     elif (pol.upper() == 'POS'):
    #         polnum = 0
    #     else:
    #         logging.error('Try to set invalid dacpolarity')

    #     start_byte = int(round((start-polnum)/4000.0*65535))
    #     stop_byte = int(round((stop-polnum)/4000.0*65535))
    #     byte_vec = np.arange(start_byte, stop_byte+1, step)
    #     mvolt_vec = byte_vec/65535.0 * 4000.0 + polnum
    #     return mvolt_vec
