import logging
from time import sleep
from functools import partial
from qcodes.instrument.visa import VisaInstrument
from qcodes.utils import validators as vals

log = logging.getLogger(__name__)


class Decadac(VisaInstrument):
    """
    The qcodes driver for the Decadac.
    Each slot on the Deacadac is to be treated as a seperate
    four-channel instrument.

    Tested with a Decadec firmware revion number 14081 (Decadac 139).

    The message strategy is the following: always keep the queue empty, so
    that self.visa_handle.ask(XXX) will return the answer to XXX and not
    some previous event.


    Attributes:

        _ramp_state (bool): If True, ramp state is ON. Default False.

        _ramp_time (int): The ramp time in ms. Default 100 ms.
    """

    def __init__(self, name, port, slot, timeout=2, baudrate=9600,
                 bytesize=8, **kwargs):

        """

        Creates an instance of the Decadac instrument corresponding to one slot
        on the physical instrument.

        Args:
            name (str): What this instrument is called locally.

            port (number): The number (only) of the COM port to connect to.

            slot (int): The slot to use.

            timeout (number): Seconds to wait for message response.
            Default 0.3.

            baudrate (number): The connection baudrate. Default 9600.

            bytesize (number): The connection bytesize. Default 8.
        """

        address = 'ASRL{:d}::INSTR'.format(port)
        self.slot = slot

        super().__init__(name, address, timeout=timeout, **kwargs)

        # set instrument operation state variables
        self._ramp_state = False
        self._ramp_time = 100
        self._voltranges = [1, 1, 1, 1]
        self._offsets = [0, 0, 0, 0]

        # channels
        for channelno in range(4):
            self.add_parameter('ch{}_voltage'.format(channelno),
                               get_cmd=partial(self._getvoltage,
                                               channel=channelno),
                               set_cmd=partial(self._setvoltage,
                                               channel=channelno),
                               label='Voltage',
                               unit='V')

            self.add_parameter('ch{}_voltrange'.format(channelno),
                               get_cmd=partial(self._getvoltrange, channelno),
                               set_cmd=partial(self._setvoltrange, channelno),
                               vals=vals.Enum(1, 2, 3))

            self.add_parameter('ch{}_offset'.format(channelno),
                               get_cmd=partial(self._getoffset, channelno),
                               set_cmd=partial(self._setoffset, channelno),
                               label='Channel {} offset'.format(channelno),
                               unit='V',
                               docstring="""
                                         The offset is applied to the channel.
                                         E.g. if ch1_offset = 1 and ch_voltage
                                         is set to 1, the instrument is told to
                                         output 2 volts.
                                         """)

        self.add_parameter('mode',
                           label='Output mode',
                           set_cmd='B {}; M {};'.format(self.slot, '{}'),
                           vals=vals.Enum(0, 1),
                           docstring="""
                                     The operational mode of the slot.
                                     0: output off, 1: output on.
                                     """)

        # initialise hardware settings
        self.mode.set(1)

    def _getoffset(self, n):
        return self._offsets[n]

    def _setoffset(self, n, val):
        self._offsets[n] = val

    def _getvoltrange(self, n):
        return self._voltranges[n]

    def _setvoltrange(self, n, val):
        self._voltranges[n] = val

    def _getvoltage(self, channel):
        """
        Function to query the voltage. Flushes the message queue in that
        process.
        """

        # set the relevant channel and slot to query
        mssg = 'B {:d}; C {:d};'.format(self.slot, channel)
        mssg += 'd;'

        # a bit of string juggling to extract the voltage
        rawresponse = self.visa_handle.ask(mssg)
        temp = rawresponse[::-1]
        temp = temp[3:temp.upper().find('D')-1]
        response = temp[::-1]

        rawvoltage = self._code2voltage(response, channel)
        actualvoltage = rawvoltage - self._offsets[channel]

        return actualvoltage

    def _setvoltage(self, voltage, channel):
        """
        Function to set the voltage. Depending on whether self._ramp_state is
        True or False, this function either ramps from the current voltage to
        the specified voltage or directly makes the voltage jump there.

        Args:
            voltage (number): the set voltage.
        """

        actualvoltage = voltage + self._offsets[channel]
        code = self._voltage2code(actualvoltage, channel)

        mssg = 'B {:d}; C {:d};'.format(self.slot, channel)

        if not self._ramp_state:
            mssg += 'D ' + code + ';'

            self.visa_handle.write(mssg)

            # due to a quirk of the Decadac, we spare the user of an error
            # sometimes encountered on first run
            try:
                self.visa_handle.read()
            except UnicodeDecodeError:
                log.warning(" Decadac returned nothing and possibly did nothing. " +
                            "Please re-run the command")
                pass

        if self._ramp_state:
            currentcode = self._voltage2code(self._getvoltage(channel),
                                             channel)
            slope = int((float(code)-float(currentcode)) /
                        (10*self._ramp_time)*2**16)
            if slope < 0:
                limit = 'L'
            else:
                limit = 'U'

            script = ['{',
                      '*1:',
                      'M2;',
                      'T 100;',  # 1 timestep: 100 micro s
                      limit + code + ';',
                      'S' + str(slope) + ';',
                      'X0;',
                      '}']
            runcmd = 'X 1;'
            mssg += ''.join(script) + runcmd
            self.visa_handle.write(mssg)
            sleep(0.0015*self._ramp_time)  # Required sleep.
            self.visa_handle.read()

            # reset channel voltage ranges
            if slope < 0:
                self.visa_handle.write('L 0;')
                self.visa_handle.read()
            else:
                self.visa_handle.write('U 65535;')
                self.visa_handle.read()

    def set_ramping(self, state, time=None):
        """
        Function to set _ramp_state and _ramp_time.

        Args:
            state (bool): True sets ramping ON.

            time (Optiona[int]): the ramp time in ms
        """
        self._ramp_state = state
        if time is not None:
            self._ramp_time = time

    def get_ramping(self):
        """
        Queries the value of self._ramp_state and self._ramp_time.

        Returns:
            str: ramp state information
        """
        switch = {True: 'ON',
                  False: 'OFF'}
        mssg = 'Ramp state: ' + switch[self._ramp_state]
        mssg += '. Ramp time: {:d} ms.'.format(int(self._ramp_time))
        return mssg

    def _code2voltage(self, code, channel):
        """
        Helper function translating a 32 bit code used internally by
        the Decadac into a voltage.

        Args:
            code (str): The code string from the instrument.

            channel (int): The relevant channel.
        """

        code = float(code)
        translationdict = {1: lambda x: (x+1)*20/2**16-10,
                           2: lambda x: (x+1)*10/2**16,
                           3: lambda x: (x+1)*10/2**16-10}

        return translationdict[self._voltranges[channel]](code)

    def _voltage2code(self, voltage, channel):
        """
        Helper function translating a voltage in V into a 32 bit code used
        internally by the Decadac.

        Args:
            voltage (float): The physical voltage.

            channel (int): The relevant channel.

        Returns:
            code (str): The corresponding voltage code.
        """
        translationdict = {1: lambda x: 2**16/20*(x-2**-16+10),
                           2: lambda x: 2**16/10*(x-2**-16),
                           3: lambda x: 2**16/10*(x-2**-16+10)}
        voltage_float = translationdict[self._voltranges[channel]](voltage)
        return str(int(voltage_float))

