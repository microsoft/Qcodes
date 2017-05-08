import time
import logging
import numpy as np
import visa  # used for the parity constant
import traceback
import threading

from qcodes import VisaInstrument, validators as vals
from qcodes.instrument.parameter import ManualParameter
from qcodes.utils.validators import Bool, Numbers


class IVVI(VisaInstrument):
    '''
    Status: Alpha version, tested for basic get-set commands
        TODO:
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
                 dac_delay=.1, dac_max_delay=0.2, safe_version=True,
                 polarity=['BIP', 'BIP', 'BIP', 'BIP'],
                 use_locks=False, **kwargs):
        '''
        Initialzes the IVVI, and communicates with the wrapper

        Args:
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
            safe_version (bool)    : if True then do not send version commands
                                     to the IVVI controller
            use_locks (bool) : if True then locks are used in the `ask`
                              function of the driver. The IVVI driver is not
                              thread safe, this locking mechanism makes it
                              thread safe at the cost of making the call to ask
                              blocking.
        '''
        t0 = time.time()
        super().__init__(name, address, **kwargs)
        if use_locks:
            self.lock = threading.Lock()
        else:
            self.lock = None

        self.safe_version = safe_version

        if numdacs % 4 == 0 and numdacs > 0:
            self._numdacs = int(numdacs)
        else:
            raise ValueError('numdacs must be a positive multiple of 4, '
                             'not {}'.format(numdacs))

        # values based on descriptor
        self.visa_handle.baud_rate = 115200
        self.visa_handle.parity = visa.constants.Parity(1)  # odd parity
        self.visa_handle.write_termination = ''
        self.visa_handle.read_termination = ''

        self.add_parameter('version',
                           get_cmd=self._get_version)

        self.add_parameter('check_setpoints',
                           parameter_class=ManualParameter,
                           initial_value=False,
                           label='Check setpoints',
                           vals=Bool(),
                           docstring=('Whether to check if the setpoint is the'
                                      ' same as the current DAC value to '
                                      'prevent an unnecessary set command.'))

        # Time to wait before sending a set DAC command to the IVVI
        self.add_parameter('dac_set_sleep',
                           parameter_class=ManualParameter,
                           initial_value=0.05,
                           label='DAC set sleep',
                           unit='s',
                           vals=Numbers(0),
                           docstring=('When check_setpoints is set to True, '
                                      'this is the waiting time between the'
                                      'command that checks the current DAC '
                                      'values and the final set DAC command'))

        # Minimum time to wait before the read buffer contains data
        self.add_parameter('dac_read_buffer_sleep',
                           parameter_class=ManualParameter,
                           initial_value=0.025,
                           label='DAC read buffer sleep',
                           unit='s',
                           vals=Numbers(0),
                           docstring=('While receiving bytes from the IVVI, '
                                      'sleeping is done in multiples of this '
                                      'value. Change to a lower value for '
                                      'a shorter minimum time to wait.'))

        self.add_parameter('dac voltages',
                           label='Dac voltages',
                           get_cmd=self._get_dacs)

        # initialize pol_num, the voltage offset due to the polarity
        self.pol_num = np.zeros(self._numdacs)
        for i in range(int(self._numdacs / 4)):
            self.set_pol_dacrack(polarity[i], np.arange(1 + i * 4, 1 + (i + 1) * 4),
                                 get_all=False)

        for i in range(1, numdacs + 1):
            self.add_parameter(
                'dac{}'.format(i),
                label='Dac {}'.format(i),
                unit='mV',
                get_cmd=self._gen_ch_get_func(self._get_dac, i),
                set_cmd=self._gen_ch_set_func(self._set_dac, i),
                vals=vals.Numbers(self.pol_num[i - 1],
                                  self.pol_num[i - 1] + self.Fullrange),
                step=dac_step,
                delay=dac_delay,
                max_delay=dac_max_delay,
                max_val_age=10)

        self._update_time = 5  # seconds
        self._time_last_update = 0  # ensures first call will always update

        t1 = time.time()

        # make sure we ignore termination characters
        # See http://www.ni.com/tutorial/4256/en/#toc2 on Termination Character
        # Enabled
        v = self.visa_handle
        v.set_visa_attribute(visa.constants.VI_ATTR_TERMCHAR_EN, 0)
        v.set_visa_attribute(visa.constants.VI_ATTR_ASRL_END_IN, 0)
        v.set_visa_attribute(visa.constants.VI_ATTR_ASRL_END_OUT, 0)
        v.set_visa_attribute(visa.constants.VI_ATTR_SEND_END_EN, 0)

        # basic test to confirm we are properly connected
        try:
            self.get_all()
        except Exception as ex:
            print('IVVI: get_all() failed, maybe connected to wrong port?')
            print(traceback.format_exc())

        print('Initialized IVVI-rack in %.2fs' % (t1 - t0))

    def get_idn(self):
        """
        Overwrites the get_idn function using constants as the hardware
        does not have a proper \*IDN function.
        """
        # not all IVVI racks support the version command, so return a dummy
        return -1

        idparts = ['QuTech', 'IVVI', 'None', self.version()]

        return dict(zip(('vendor', 'model', 'serial', 'firmware'), idparts))

    def _get_version(self):
        if self.safe_version:
            return -1
        else:
            # ask for the version of more recent modules
            # some of the older modules cannot handle this command
            mes = self.ask(bytes([3, 4]))
            ver = mes[2]
            return ver

    def get_all(self):
        return self.snapshot(update=True)

    def set_dacs_zero(self):
        for i in range(self._numdacs):
            self.set('dac{}'.format(i + 1), 0)

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
        bytevalue = int(round(mvoltage / self.Fullrange * 65535))
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
            values[i] = ((byte_mess[2 + 2 * i] * 256 + byte_mess[3 + 2 * i]) /
                         65535.0 * self.Fullrange) + self.pol_num[i]
        return values

    # Communication with device
    def _get_dac(self, channel):
        '''
        Returns dac channel in mV
        channels range from 1-numdacs

        this version is a wrapper around the IVVI get function.
        it only updates
        '''
        return self._get_dacs()[channel - 1]

    def _set_dac(self, channel, mvoltage):
        '''
        Sets the specified dac to the specified voltage.
        A check to prevent setting the same value is performed if
        the check_setpoints flag was set.

        Input:
            mvoltage (float) : output voltage in mV
            channel (int)    : 1 based index of the dac

        Output:
            reply (string) : errormessage
        Private version of function
        '''
        proceed = True

        if self.check_setpoints():
            cur_val = self.get('dac{}'.format(channel))
            # dac range in mV / 16 bits FIXME make range depend on polarity
            byte_res = self.Fullrange / 2**16
            # eps is a magic number to correct for an offset in the values
            # the IVVI returns (i.e. setting 0 returns byte_res/2 = 0.030518
            # with rounding
            eps = 0.0001

            proceed = False

            if (mvoltage > (cur_val + byte_res / 2 + eps) or
                    mvoltage < (cur_val - byte_res / 2 - eps)):
                proceed = True

            if self.dac_set_sleep() > 0.0:
                time.sleep(self.dac_set_sleep())

        # only update the value if it is different from the previous one
        # this saves time in setting values, set cmd takes ~650ms
        if proceed:
            polarity_corrected = mvoltage - self.pol_num[channel - 1]
            byte_val = self._mvoltage_to_bytes(polarity_corrected)
            message = bytes([2, 1, channel]) + byte_val

            reply = self.ask(message)
            self._time_last_update = 0  # ensures get command will update

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
        if (time.time() - self._time_last_update) > self._update_time:
            message = bytes([self._numdacs * 2 + 2, 2])
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
            if i + 1 == max_tries:  # +1 because range goes stops before end
                raise ex
        return self._mvoltages

    def write(self, message, raw=False):
        '''
        Protocol specifies that a write consists of
        descriptor size, error_code, message

        returns message_len
        '''
        # This is used when write is used in the ask command
        expected_answer_length = None

        if not raw:
            expected_answer_length = message[0]
            message_len = len(message) + 2

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
        if self.lock:
            max_tries = 10
            for i in range(max_tries):
                if self.lock.acquire(timeout=.05):
                    break
                else:
                    logging.warning('IVVI: cannot acquire the lock')
            if i + 1 == max_tries:
                raise Exception('IVVI: lock is stuck')
        # Protocol knows about the expected length of the answer
        message_len = self.write(message, raw=raw)
        reply = self.read(message_len=message_len)
        if self.lock:
            self.lock.release()

        return reply

    def _read_raw_bytes_direct(self, size):
        """ Read raw data using the visa lib """
        with(self.visa_handle.ignore_warning(visa.constants.VI_SUCCESS_MAX_CNT)):
            data, statuscode = self.visa_handle.visalib.read(
                self.visa_handle.session, size)

        return data

    def _read_raw_bytes_multiple(self, size, maxread=512, verbose=0):
        """ Read raw data in blocks using the visa lib
        Arguments:
            size (int) : number of bytes to read
            maxread (int) : maximum size of block to read
            verbose (int): verbosity level
        Returns:
            ret (bytes): bytes read from the device
        The pyvisa visalib.read does not always terminate at a newline, this
        is a workaround.
        Also see: https://github.com/qdev-dk/Qcodes/issues/276
                  https://github.com/hgrecco/pyvisa/issues/225
        Setting both VI_ATTR_TERMCHAR_EN and VI_ATTR_ASRL_END_IN to zero
        should allow the driver to ignore termination characters, this
        function is an additional safety mechanism.
        """
        ret = []
        instr = self.visa_handle
        with self.visa_handle.ignore_warning(visa.constants.VI_SUCCESS_MAX_CNT):
            nread = 0
            while nread < size:
                nn = min(maxread, size - nread)
                chunk, status = instr.visalib.read(instr.session, nn)
                ret += [chunk]
                nread += len(chunk)
                if verbose:
                    print('_read_raw: %d/%d bytes' % (len(chunk), nread))
        ret = b''.join(ret)
        return ret

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

            if self.dac_read_buffer_sleep() > 0.0:
                time.sleep(self.dac_read_buffer_sleep())

            bytes_in_buffer = self.visa_handle.bytes_in_buffer
            if t1 - t0 > timeout:
                raise TimeoutError()
        # a workaround for a timeout error in the pyvsia read_raw() function
        mes = self._read_raw_bytes_multiple(bytes_in_buffer)

        # if mes[1] != 0:
        # see protocol descriptor for error codes
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
            self.pol_num[ch - 1] = val
            # self.set_parameter_bounds('dac%d' % (i+1), val, val +
            # self.Fullrange.0)

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
        val = self.pol_num[channel - 1]

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

    def adjust_parameter_validator(self, param):
        """Adjust the parameter validator range based on the dac resolution.

        The dac's of the IVVI have a finite resolution. If the validator range
        min and max values are not values the dac's can actually have, then it
        can occur that a set command results in the dac's going to a value just
        outside the validator range. Adjusting the validators with this
        function prevents that.

        Args:
            param (StandardParameter): a dac of the IVVI instrument
        """
        if type(param._vals) is not Numbers:
            raise Exception('Only the Numbers validator is supported.')
        min_val = param._vals._min_value
        max_val = param._vals._max_value
        min_val_pol_corr = min_val - self.pol_num[int(param.name[3:]) - 1]
        max_val_pol_corr = max_val - self.pol_num[int(param.name[3:]) - 1]
        min_val_bytes = self._mvoltage_to_bytes(min_val_pol_corr)
        min_val_upd = (min_val_bytes[0] * 256 + min_val_bytes[1]) / \
            65535.0 * self.Fullrange + self.pol_num[int(param.name[3:]) - 1]
        max_val_bytes = self._mvoltage_to_bytes(max_val_pol_corr)
        max_val_upd = (max_val_bytes[0] * 256 + max_val_bytes[1]) / \
            65535.0 * self.Fullrange + self.pol_num[int(param.name[3:]) - 1]
        param._vals = Numbers(min_val_upd, max_val_upd)

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
