from qcodes.instrument.visa import VisaInstrument
from time import sleep
from functools import partial


class Decadac(VisaInstrument):
    """
    The qcodes driver for the Decadac.
    Each slot on the Deacadac is to be treated as a seperate
    four-channel instrument.

    Tested with a Decadec firmware revion number 14081 (Decadac 139).

    The message strategy is the following: always keep the queue empty, so
    that self.visa_handle.ask(XXX) will return the answer to XXX and not
    some previous event.

    The class comes with the following methods and attributes:

    Methods:

        set_ramping(state:bool, time:float): a shortcut to set
            self.ramp_state and self.ramp_time

        get_ramping: Queries the value of self.ramp_state and
            self.ramp_time. Returns a string.

    Attributes:

        ramp_state (bool): If True, ramp state is ON. Default False.

        ramp_time (num): The ramp time in ms. Default 100 ms.

        voltranges (list): The voltranges for each channel. Can be read and
        set (using a screwdriver) on the front of the physical instrument.
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
        self.ramp_state = False
        self.ramp_time = 100

        # initialise hardware settings
        self.voltranges = [1, 1, 1, 1]

        # set up four channels as qcodes parameters
        for channelno in range(4):

            self.add_parameter('volt{:d}'.format(channelno),
                               get_cmd=partial(self._getvoltage,
                                               channel=channelno),
                               set_cmd=partial(self._setvoltage,
                                               channel=channelno),
                               label='Voltage',
                               units='V')

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

        return self._code2voltage(response, channel)

    def _setvoltage(self, voltage, channel):
        """
        Function to set the voltage. Depending on whether self.ramp_state is
        True or False, this function either ramps from the current voltage to
        the specified voltage or directly makes the voltage jump there.

        Args:
            voltage (number): the set voltage.
        """

        code = self._voltage2code(voltage, channel)

        mssg = 'B {:d}; C {:d};'.format(self.slot, channel)

        if not self.ramp_state:
            mssg += 'D ' + code + ';'

            self.visa_handle.write(mssg)

            # due to a quirk of the Decadac, we spare the user of an error
            # sometimes encountered on first run
            try:
                self.visa_handle.read()
            except UnicodeDecodeError:
                pass

        if self.ramp_state:
            currentcode = self._voltage2code(self._getvoltage(channel),
                                             channel)
            slope = int((float(code)-float(currentcode)) /
                        (10*self.ramp_time)*2**16)
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
            sleep(0.0015*self.ramp_time)  # Required sleep.
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
        Function to set ramp_state and ramp_time.

        Args:
            state (bool): True sets ramping ON.

            time (int): the ramp time in ms
        """
        self.ramp_state = state
        if time is not None:
            self.ramp_time = time

    def get_ramping(self):
        """Returns a string with ramp state information"""
        switch = {True: 'ON',
                  False: 'OFF'}
        mssg = 'Ramp state: ' + switch[self.ramp_state]
        mssg += '. Ramp time: {:d} ms.'.format(int(self.ramp_time))
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

        return translationdict[self.voltranges[channel]](code)

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
        return str(int(translationdict[self.voltranges[channel]](voltage)))

