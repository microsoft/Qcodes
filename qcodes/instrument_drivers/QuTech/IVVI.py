import time
import logging
import numpy as np
import visa  # used for the parity constant
import traceback

from qcodes import VisaInstrument, validators as vals


class IVVI(VisaInstrument):
    '''
    Status: Alpha version, tested for basic get-set commands
        TODO:
            - Add individual parameters for channel polarities
            - Test polarities different from BIP
            - Add adjustable range and rate protection per channel
            - Add error handling for the specific error messages in the
              protocol
            - Remove/fine-tune manual sleep statements

    This is the python driver for the D5 module of the IVVI-rack
    see: http://qtwork.tudelft.nl/~schouten/ivvi/doc-d5/index-d5.htm

    A descriptor for the data protocol can be found at
    http://qtwork.tudelft.nl/~schouten/ivvi/doc-d5/rs232linkformat.txt
    A copy of this file can be found at the bottom of this file.
    '''
    Fullrange = 4000
    Halfrange = Fullrange / 2

    def __init__(self, name, address, reset=False, numdacs=16, dac_step=10,
                 dac_delay=.1, dac_max_delay=0.2, **kwargs):
                 # polarity=['BIP', 'BIP', 'BIP', 'BIP']):
                 # commented because still on the todo list
        '''
        Initialzes the IVVI, and communicates with the wrapper
        Input:
            name (string)        : name of the instrument
            address (string)     : ASRL address
            reset (bool)         : resets to default values, default=false
            numdacs (int)        : number of dacs, multiple of 4, default=16
            polarity (string[4]) : list of polarities of each set of 4 dacs
                                   choose from 'BIP', 'POS', 'NEG',
                                   default=['BIP', 'BIP', 'BIP', 'BIP']
            dac_step (float)         : max step size for dac parameter
            dac_delay (float)        : delay (in seconds) for dac
            dac_max_delay (float)    : maximum delay before emitting a warning
        '''
        t0 = time.time()
        super().__init__(name, address, **kwargs)

        if numdacs % 4 == 0 and numdacs > 0:
            self._numdacs = int(numdacs)
        else:
            raise ValueError('numdacs must be a positive multiple of 4, '
                             'not {}'.format(numdacs))

        # values based on descriptor
        self.visa_handle.baud_rate = 115200
        self.visa_handle.parity = visa.constants.Parity(1)  # odd parity

        self.add_parameter('version',
                           get_cmd=self._get_version)
        
        self.add_parameter('dac voltages',
                           label='Dac voltages',
                           get_cmd=self._get_dacs)

        for i in range(1, numdacs + 1):
            self.add_parameter(
                'dac{}'.format(i),
                label='Dac {} (mV)'.format(i),
                units='mV',
                get_cmd=self._gen_ch_get_func(self._get_dac, i),
                set_cmd=self._gen_ch_set_func(self._set_dac, i),
                vals=vals.Numbers(-2000, 2000),
                step=dac_step,
                delay=dac_delay,
                max_delay=dac_max_delay,
                max_val_age=10)

        self._update_time = 5  # seconds
        self._time_last_update = 0  # ensures first call will always update
        
        self.pol_num = np.zeros(self._numdacs)  # corresponds to POS polarity
        self.set_pol_dacrack('BIP', range(self._numdacs), get_all=False)

        t1 = time.time()

        # basic test to confirm we are properly connected
        try:
            self.get_all()
        except Exception as ex:
            print('IVVI: get_all() failed, maybe connected to wrong port?')
            print(traceback.format_exc())

        print('Initialized IVVI-rack in %.2fs' % (t1-t0))

    def get_idn(self):
        """
        Overwrites the get_idn function using constants as the hardware
        does not have a proper *IDN function.
        """
        idparts = ['QuTech', 'IVVI', 'None', self.version()]

        return dict(zip(('vendor', 'model', 'serial', 'firmware'), idparts))

    def _get_version(self):
        mes = self.ask(bytes([3, 4]))
        v = mes[2]
        return v

    def get_all(self):
        return self.snapshot(update=True)

    def set_dacs_zero(self):
        for i in range(self._numdacs):
            self._set_dac(i+1, 0)

    # Conversion of data
    def _mvoltage_to_bytes(self, mvoltage):
        '''
        Converts a mvoltage on a 0mV-4000mV scale to a 16-bit integer
        equivalent

        output is a list of two bytes

        Input:
            mvoltage (float) : a mvoltage in the 0mV-4000mV range

        Output:
            (dataH, dataL) (int, int) : The high and low value byte equivalent
        '''
        bytevalue = int(round(mvoltage/self.Fullrange*65535))
        return bytevalue.to_bytes(length=2, byteorder='big')

    def _bytes_to_mvoltages(self, byte_mess):
        '''
        Converts a list of bytes to a list containing
        the corresponding mvoltages
        '''
        values = list(range(self._numdacs))
        for i in range(self._numdacs):
            # takes two bytes, converts it to a 16 bit int and then divides by
            # the range and adds the offset due to the polarity
            values[i] = ((byte_mess[2 + 2*i]*256 + byte_mess[3 + 2*i]) /
                         65535.0*self.Fullrange) + self.pol_num[i]
        return values

    # Communication with device
    def _get_dac(self, channel):
        '''
        Returns dac channel in mV
        channels range from 1-numdacs

        this version is a wrapper around the IVVI get function.
        it only updates
        '''
        return self._get_dacs()[channel-1]

    def _set_dac(self, channel, mvoltage):
        '''
        Sets the specified dac to the specified voltage.
        Will only send a command to the IVVI if the next value is different
        than the current value within byte resolution.

        Input:
            mvoltage (float) : output voltage in mV
            channel (int)    : 1 based index of the dac

        Output:
            reply (string) : errormessage
        Private version of function
        '''
        cur_val = self.get('dac{}'.format(channel))
        # dac range in mV / 16 bits FIXME make range depend on polarity
        byte_res = self.Fullrange/2**16
        eps = 0.0001
        # eps is a magic number to correct for an offset in the values the IVVI
        # returns (i.e. setting 0 returns byte_res/2 = 0.030518 with rounding

        # only update the value if it is different from the previous one
        # this saves time in setting values, set cmd takes ~650ms
        if (mvoltage > (cur_val+byte_res/2+eps) or
                mvoltage < (cur_val - byte_res/2-eps)):
            byte_val = self._mvoltage_to_bytes(mvoltage -
                                               self.pol_num[channel-1])
            message = bytes([2, 1, channel]) + byte_val
            time.sleep(.05)
            reply = self.ask(message)
            self._time_last_update = 0  # ensures get command will update
            return reply
        return

    def _get_dacs(self):
        '''
        Reads from device and returns all dacvoltages in a list

        Input:
            None

        Output:
            voltages (float[]) : list containing all dacvoltages (in mV)

        get dacs command takes ~450ms according to ipython timeit
        '''
        if (time.time() - self._time_last_update) > self._update_time:
            message = bytes([self._numdacs*2+2, 2])
            # workaround for an error in the readout that occurs sometimes
            max_tries = 10
            for i in range(max_tries):
                try:
                    reply = self.ask(message)
                    self._mvoltages = self._bytes_to_mvoltages(reply)
                    self._time_last_update = time.time()
                    break
                except Exception as ex:
                    logging.warning('IVVI communication error trying again')
            if i+1 == max_tries:  # +1 because range goes stops before end
                raise('IVVI Communication error')
        return self._mvoltages

    def write(self, message, raw=False):
        '''
        Protocol specifies that a write consists of
        descriptor size, error_code, message

        returns message_len
        '''
        # This is used when write is used in the ask command
        expected_answer_length = message[0]
        if not raw:
            message_len = len(message)+2
            error_code = bytes([0])
            message = bytes([message_len]) + error_code + message
        self.visa_handle.write_raw(message)

        return expected_answer_length

    def ask(self, message, raw=False):
        '''
        Send <message> to the device and read answer.
        Raises an error if one occurred
        Returns a list of bytes
        '''
        # Protocol knows about the expected length of the answer
        message_len = self.write(message, raw=raw)
        return self.read(message_len=message_len)

    def read(self, message_len=None):
        # because protocol has no termination chars the read reads the number
        # of bytes in the buffer
        bytes_in_buffer = 0
        timeout = 1
        t0 = time.time()
        t1 = t0
        bytes_in_buffer = 0
        if message_len is None:
            message_len = 1  # ensures at least 1 byte in buffer

        while bytes_in_buffer < message_len:
            t1 = time.time()
            time.sleep(.05)
            bytes_in_buffer = self.visa_handle.bytes_in_buffer
            if t1-t0 > timeout:
                raise TimeoutError()
        # a workaround for a timeout error in the pyvsia read_raw() function
        with(self.visa_handle.ignore_warning(visa.constants.VI_SUCCESS_MAX_CNT)):
            mes = self.visa_handle.visalib.read(
                self.visa_handle.session, bytes_in_buffer)
        mes = mes[0]  # cannot be done on same line for some reason
        # if mes[1] != 0:
        #     # see protocol descriptor for error codes
        #     raise Exception('IVVI rack exception "%s"' % mes[1])
        return mes

    def set_pol_dacrack(self, flag, channels, get_all=True):
        '''
        Changes the polarity of the specified set of dacs

        Input:
            flag (string) : 'BIP', 'POS' or 'NEG'
            channel (int) : 0 based index of the rack
            get_all (boolean): if True (default) perform a get_all

        Output:
            None
        '''
        flagmap = {'NEG': -self.Fullrange, 'BIP': -self.Halfrange, 'POS': 0}
        if flag.upper() not in flagmap:
            raise KeyError('Tried to set invalid dac polarity %s', flag)

        val = flagmap[flag.upper()]
        for ch in channels:
            self.pol_num[ch-1] = val
            # self.set_parameter_bounds('dac%d' % (i+1), val, val + self.Fullrange.0)

        if get_all:
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

        if (val == -self.Fullrange):
            return 'NEG'
        elif (val == -self.Halfrange):
            return 'BIP'
        elif (val == 0):
            return 'POS'
        else:
            return 'Invalid polarity in memory'

    def _gen_ch_set_func(self, fun, ch):
        def set_func(val):
            return fun(ch, val)
        return set_func

    def _gen_ch_get_func(self, fun, ch):
        def get_func():
            return fun(ch)
        return get_func

'''
RS232 PROTOCOL
-----------------------
BAUTRATE    115200
DATA BITS   8
PARITY      ODD
STOPBITS    1

Descriptor data PC-> MC

Byte        Name               Description                              value
--------------------------------------------------------------------------------------------------------
0        Descriptor size         Size of this descriptor                4 (action 2,4,6,7)
                                                                        5 (action 7)
                                                                        7 (action 1,3)
                                                                        11 (action 5)
1        Error                                                          0
2        Data out size           Number of bytes that has to be         2 (action 0,1,3,5,6)
                                 send by the MC after receiving         3 (action 4)
                                 descriptor                             4 (action 7)
                                                                        34 (action 2)
3        Action                                                         0= no operation
                                                                        1= set Dac value
                                                                        2= request DAC data
                                                                        3= continues send data to DAC
                                                                        4= ask for Program  ion
                                                                        5= set bits interface
                                                                        6= generate trigger output
                                                                        7= request data from specified DAC

4        Dac nr                  Nr of DAC to be updated                1 to 16
5        DataH                   High byte to DAC                       0 to $ff
6        DataL                   Low byte to DAC                        0 to $ff
7        data bit 24-31          interfaceBit24_31                      0 to $ff
8        data bit 16-23          interfaceBit16_23                      0 to $ff
9        data bit 08-15          interfaceBit08_15                      0 to $ff
10       data bit 00-07          interfaceBit00_07                      0 to $ff
--------------------------------------------------------------------------------------------------------


Descriptor data MC-> PC
--------------------------------------------------------------------------------------------------------
0        Descriptor size         Size of this descriptor                1
1        Error                   0x00 = no Error detected               1
                                 0x01 =
                                 0x02 =
                                 0x04 = Parity
                                 0x08 = Overrun
                                 0x10 = Frame Error
                                 0x20 = WatchDog reset detected (32)
                                 0x40 = DAC does not exist(64)
                                 0x80 = WrongAction (128)
2        Version                 program version                       1
2        DAC1                    Value of DAC1                         2
4        DAC2                    Value of DAC2                         2
..
32       DAC16                   Value of DAC16                        2
--------------------------------------------------------------------------------------------------------
'''