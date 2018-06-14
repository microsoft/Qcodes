from time import time
from functools import partial
from qcodes import VisaInstrument, InstrumentChannel, ChannelList
from qcodes.utils import validators as vals


class DACException(Exception):
    pass


class DacReader(object):
    @staticmethod
    def _dac_parse(resp):
        """
        Parses responses from the DAC. They should take the form of "<cmd><resp>!"
        This command returns the value of resp.
        """
        resp = resp.strip()
        if resp[-1] != "!":
            raise DACException("Unexpected terminator on response: {}. Should end with \"!\"".format(resp))
        return resp.strip()[1:-1]

    def _dac_v_to_code(self, volt):
        """
        Convert a voltage to the internal dac code (number between 0-65536)
        based on the minimum/maximum values of a given channel.
        Midrange is 32768.
        """
        if volt < self.min_val or volt >= self.max_val:
            raise ValueError('Cannot convert voltage {} V '.format(volt) +
                             'to a voltage code, value out of range '
                             '({} V - {} V).'.format(self.min_val,
                                                     self.max_val))

        frac = (volt - self.min_val) / (self.max_val - self.min_val)
        val = int(round(frac * 65536))
        # extra check to be absolutely sure that the instrument does nothing
        # receive an out-of-bounds value
        if val > 65535 or val < 0:
            raise ValueError('Voltage ({} V) resulted in the voltage code {}'
                             ', which is not within the allowed range.'
                             ''.format(volt, val))
        return val

    def _dac_code_to_v(self, code):
        """
        Convert a voltage to the internal dac code (number between 0-65536)
        based on the minimum/maximum values of a given channel.
        Midrange is 32768.
        """
        frac = code/65536.0
        return (frac * (self.max_val - self.min_val)) + self.min_val

    def _set_slot(self):
        """
        Set the active DAC slot
        """
        resp = self.ask_raw("B{};".format(self._slot))
        if int(self._dac_parse(resp)) != self._slot:
            raise DACException("Unexpected return from DAC when setting slot: "
                               "{}. DAC slot may not have been set.".format(resp))

    def _set_channel(self):
        """
        Set the active DAC channel
        """
        resp = self.ask_raw("B{};C{};".format(self._slot, self._channel))
        if resp.strip() != "B{}!C{}!".format(self._slot, self._channel):
            raise DACException("Unexpected return from DAC when setting channel: "
                               "{}. DAC channel may not have been set.".format(resp))

    def _query_address(self, addr, count=1, versa_eeprom=False):
        """
        Query the value at the dac address given.

        Args:
            addr (int): The address to query.

            count (int): The number of bytes to query.

            versa_eeprom(bool): do we want to read from the versadac (slot) EEPROM
        """
        # Check if we actually have anything to query
        if count == 0:
            return 0

        # Validate address
        addr = int(addr)
        if addr < 0 or addr > 1107296266:
            raise DACException("Invalid address {}.".format(addr))

        # Choose a poke command depending on whether we are querying a
        # VERSADAC eeprom or main memory
        # If we are writing to a VERSADAC, we must also set the slot.
        if versa_eeprom:
            self._set_slot()
            query_command = "e;"
        else:
            query_command = "p;"

        # Read a number of bytes from the device and convert to an int
        val = 0
        for i in range(count):
            # Set DAC to point to address
            ret = int(self._dac_parse(self.ask_raw("A{};".format(addr))))
            if ret != addr:
                raise DACException("Failed to set EEPROM address {}.".format(addr))
            val += int(self._dac_parse(self.ask_raw(query_command))) << (32*(count-i-1))
            addr += 1

        return val

    def _write_address(self, addr, val, versa_eeprom=False):
        """
        Write a value to a given DAC address

        Args:
            addr (int): The address to query.

            val (int): The value to write.

            versa_eeprom(bool): do we want to read from the versadac (slot) EEPROM
        """
        # Validate address
        addr = int(addr)
        if addr < 0 or addr > 1107296266:
            raise DACException("Invalid address {}.".format(addr))

        # Validate value
        val = int(val)
        if val < 0 or val >= 2**32:
            raise DACException("Writing invalid value ({}) to address {}.".format(val, addr))

        # Choose a poke command depending on whether we are querying a
        # VERSADAC eeprom or main memory. If we are writing to a versadac channel
        # we must also set the slot
        if versa_eeprom:
            query_command = "e;"
            write_command = "E"
            self._set_slot()
        else:
            query_command = "p;"
            write_command = "P"

        # Write the value to the DAC
        # Set DAC to point to address
        ret = int(self._dac_parse(self.ask_raw("A{};".format(addr))))
        if ret != addr:
            raise DACException("Failed to set EEPROM address {}.".format(addr))
        self.ask_raw("{}{};".format(write_command, val))
        # Check the write was successful
        if int(self._dac_parse(self.ask_raw(query_command))) != val:
            raise DACException("Failed to write value ({}) to address {}.".format(val, addr))


class DacChannel(InstrumentChannel, DacReader):
    """
    A single DAC channel of the DECADAC
    """
    _CHANNEL_VAL = vals.Ints(0, 3)

    def __init__(self, parent, name, channel, min_val=-5, max_val=5):
        super().__init__(parent, name)

        # Validate slot and channel values
        self._CHANNEL_VAL.validate(channel)
        self._channel = channel
        self._slot = self._parent._slot

        # Calculate base address for querying channel parameters
        # Note that the following values can be found using these offsets
        # 0: Interrupt Period
        # 4: DAC High Limit
        # 5: DAC Low Limit
        # 6: Slope (double)
        # 8: DAC Value (double)
        self._base_addr = 1536 + (16*4)*self._slot + 16*self._channel

        # Store min/max voltages
        assert(min_val < max_val)
        self.min_val = min_val
        self.max_val = max_val

        # Add channel parameters
        # Note we will use the older addresses to read the value from the dac rather than the newer
        # 'd' command for backwards compatibility
        self._volt_val = vals.Numbers(self.min_val, self.max_val)
        self.add_parameter("volt", get_cmd=partial(self._query_address, self._base_addr+9, 1),
                           get_parser=self._dac_code_to_v,
                           set_cmd=self._set_dac, set_parser=self._dac_v_to_code, vals=self._volt_val,
                           label="channel {}".format(channel+self._slot*4), unit="V")
        # The limit commands are used to sweep dac voltages. They are not safety features.
        self.add_parameter("lower_ramp_limit", get_cmd=partial(self._query_address, self._base_addr+5),
                           get_parser=self._dac_code_to_v,
                           set_cmd="L{};", set_parser=self._dac_v_to_code, vals=self._volt_val,
                           label="Lower_Ramp_Limit", unit="V")
        self.add_parameter("upper_ramp_limit", get_cmd=partial(self._query_address, self._base_addr+4),
                           get_parser=self._dac_code_to_v,
                           set_cmd="U{};", set_parser=self._dac_v_to_code, vals=self._volt_val,
                           label="Upper_Ramp_Limit", unit="V")
        self.add_parameter("update_period", get_cmd=partial(self._query_address, self._base_addr),
                           get_parser=int, set_cmd="T{};", set_parser=int, vals=vals.Ints(50, 65535),
                           label="Update_Period", unit="us")
        self.add_parameter("slope", get_cmd=partial(self._query_address, self._base_addr+6, 2),
                           get_parser=int, set_cmd="S{};", set_parser=int, vals=vals.Ints(-(2**32), 2**32),
                           label="Ramp_Slope")

        # Manual parameters to control whether DAC channels should ramp to voltages or jump
        self._ramp_val = vals.Numbers(0, 10)
        self.add_parameter("enable_ramp", get_cmd=None, set_cmd=None, initial_value=False,
                           vals=vals.Bool())
        self.add_parameter("ramp_rate", get_cmd=None, set_cmd=None, initial_value=0.1,
                           vals=self._ramp_val, unit="V/s")

        # Add ramp function to the list of functions
        self.add_function("ramp", call_cmd=self._ramp, args=(self._volt_val, self._ramp_val))

        # If we have access to the VERSADAC (slot) EEPROM, we can set the inital
        # value of the channel.
        # NOTE: these values will be overwritten by a K3 calibration
        if self._parent._VERSA_EEPROM_available:
            _INITIAL_ADDR = [6, 8, 32774, 32776]
            self.add_parameter("initial_value",
                               get_cmd=partial(self._query_address, _INITIAL_ADDR[self._channel], versa_eeprom=True),
                               get_parser=self._dac_code_to_v,
                               set_cmd=partial(self._write_address, _INITIAL_ADDR[self._channel], versa_eeprom=True),
                               set_parser=self._dac_v_to_code, vals=vals.Numbers(self.min_val, self.max_val))

    def _ramp(self, val, rate, block=True):
        """
        Ramp the DAC to a given voltage.

        Params:
            val (float): The voltage to ramp to in volts

            rate (float): The ramp rate in units of volts/s

            block (bool): Should the call block until the ramp is complete?
        """

        # We need to know the current dac value (in raw units), as well as the update rate
        c_volt = self.volt.get()  # Current Voltage
        if c_volt == val:  # If we are already at the right voltage, we don't need to ramp
            return
        c_val = self._dac_v_to_code(c_volt)  # Current voltage in DAC units
        e_val = self._dac_v_to_code(val)  # Endpoint in DAC units
        t_rate = 1/(self.update_period.get() * 1e-6)  # Number of refreshes per second
        secs = abs((c_volt - val)/rate)  # Number of seconds to ramp

        # The formula to calculate the slope is: Number of DAC steps divided by the number of time
        # steps in the ramp multiplied by 65536
        slope = int(((e_val - c_val)/(t_rate*secs))*65536)

        # Now let's set up our limits and ramo slope
        if slope > 0:
            self.upper_ramp_limit.set(val)
        else:
            self.lower_ramp_limit.set(val)
        self.slope.set(slope)

        # Block until the ramp is complete is block is True
        if block:
            while self.slope.get() != 0:
                pass

    def _set_dac(self, code):
        """
        Set the voltage on the dac channel, ramping if the enable_rate
        parameter is set for this channel.

        Params:
            code (int): the DAC code to set the voltage to
        """
        if self.enable_ramp.get():
            self._ramp(self._dac_code_to_v(code), rate=self.ramp_rate.get())
        else:
            code = int(code)
            self._set_channel()
            self.ask_raw("U65535;L0;D{};".format(code))

    def write(self, cmd):
        """
        Overload write to set channel prior to any channel operations.
        Since all commands are echoed back, we must keep track of responses
        as well, otherwise commands receive the wrong response.
        """
        self._set_channel()
        return self.ask_raw(cmd)

    def ask(self, cmd):
        """
        Overload ask to set channel prior to operations
        """
        self._set_channel()
        return self.ask_raw(cmd)


class DacSlot(InstrumentChannel, DacReader):
    """
    A single DAC Slot of the DECADAC
    """
    _SLOT_VAL = vals.Ints(0, 4)
    SLOT_MODE_DEFAULT = "Coarse"

    def __init__(self, parent, name, slot, min_val=-5, max_val=5):
        super().__init__(parent, name)

        # Validate slot and channel values
        self._SLOT_VAL.validate(slot)
        self._slot = slot

        # Store whether we have access to the VERSADAC EEPROM
        self._VERSA_EEPROM_available = self._parent._VERSA_EEPROM_available

        # Create a list of channels in the slot
        channels = ChannelList(self, "Slot_Channels", parent.DAC_CHANNEL_CLASS)
        for i in range(4):
            channels.append(parent.DAC_CHANNEL_CLASS(self, "Chan{}".format(i), i,
                                       min_val=min_val, max_val=max_val))
        self.add_submodule("channels", channels)
        # Set the slot mode. Valid modes are:
        #   Off: Channel outputs are disconnected from the input, grounded with 10MOhm.
        #   Fine: 2-channel mode. Channels 0 and 1 are output, use 2 and 3 for fine
        #       adjustment of Channels 0 and 1 respectively
        #   Coarse: All 4 channels are used as output
        #   FineCald: Calibrated 2-channel mode, with 0 and 1 output, 2 and 3 used
        #       automatically for fine adjustment. This mode only works for calibrated
        #       DecaDAC's
        # Unfortunately there is no known way of reading the slot mode hence this will be
        # set in initialization
        if self._parent._cal_supported:
            slot_modes = {"Off": 0, "Fine": 1, "Coarse": 2, "FineCald": 3}
        else:
            slot_modes = {"Off": 0, "Fine": 1, "Coarse": 2}
        self.add_parameter('slot_mode', get_cmd="m;", get_parser=self._dac_parse, set_cmd="M{};",
                           val_mapping=slot_modes)

        # Enable all slots in coarse mode.
        self.slot_mode.set(self.SLOT_MODE_DEFAULT)

    def write(self, cmd):
        """
        Overload write to set channel prior to any channel operations.
        Since all commands are echoed back, we must keep track of responses
        as well, otherwise commands receive the wrong response.
        """
        self._set_slot()
        return self.ask_raw(cmd)

    def ask(self, cmd):
        """
        Overload ask to set channel prior to operations
        """
        self._set_slot()
        return self.ask_raw(cmd)


class Decadac(VisaInstrument, DacReader):
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
    DAC_CHANNEL_CLASS = DacChannel
    DAC_SLOT_CLASS = DacSlot

    def __init__(self, name, address, min_val=-5, max_val=5, **kwargs):
        """

        Creates an instance of the Decadac instrument corresponding to one slot
        on the physical instrument.

        Args:
            name (str): What this instrument is called locally.

            port (str): The address of the DAC. For a serial port this is ASRLn::INSTR
                where n is replaced with the address set in the VISA control panel.
                Baud rate and other serial parameters must also be set in the VISA control
                panel.

            min_val (number): The minimum value in volts that can be output by the DAC.
                This value should correspond to the DAC code 0.

            max_val (number): The maximum value in volts that can be output by the DAC.
                This value should correspond to the DAC code 65536.

        """

        super().__init__(name, address, **kwargs)

        # Do feature detection
        self._feature_detect()

        # Create channels
        channels = ChannelList(self, "Channels", self.DAC_CHANNEL_CLASS, snapshotable=False)
        slots = ChannelList(self, "Slots", self.DAC_SLOT_CLASS)
        for i in range(5):  # Create the 6 DAC slots
            slots.append(self.DAC_SLOT_CLASS(self, "Slot{}".format(i), i, min_val, max_val))
            channels.extend(slots[i].channels)
        slots.lock()
        channels.lock()
        self.add_submodule("slots", slots)
        self.add_submodule("channels", channels)

        self.connect_message()

    def set_all(self, volt):
        """
        Set all dac channels to a specific voltage. If channels are set to ramp then the ramps
        will occur in sequence, not simultaneously.

        Args:
            volt(float): The voltage to set all gates to.
        """
        for chan in self.channels:
            chan.volt.set(volt)

    def ramp_all(self, volt, ramp_rate):
        """
        Ramp all dac channels to a specific voltage at the given rate simultaneously. Note
        that the ramps are not synchronized due to communications time and DAC ramps starting
        as soon as the commands are in.

        Args:
            volt(float): The voltage to ramp all channels to.

            ramp_rate(float): The rate in volts per second to ramp
        """
        # Start all channels ramping
        for chan in self.channels:
            chan._ramp(volt, ramp_rate, block=False)

        # Wait for all channels to complete ramping.
        # The slope is reset to 0 once ramping is complete.
        for chan in self.channels:
            while chan.slope.get():
                pass

    def get_idn(self):
        """
        Attempt to identify the dac. Since we don't have standard SCPI commands, ``*IDN`` will
        do nothing on this DAC.

        Returns:
            A dict containing a serial and hardware version
        """
        self._feature_detect()

        return {"serial": self.serial_no, "hardware_version": self.version}

    def connect_message(self, idn_param='IDN', begin_time=None):
        """
        Print a connect message, taking into account the lack of a standard ``*IDN`` on
        the Harvard DAC

        Args:
            begin_time (number): time.time() when init started.
                Default is self._t0, set at start of Instrument.__init__.
        """
        # start with an empty dict, just in case an instrument doesn't
        # heed our request to return all 4 fields.
        t = time() - (begin_time or self._t0)

        con_msg = ("Connected to Harvard DecaDAC "
                   "(hw ver: {}, serial: {}) in {:.2f}s".format(
                    self.version, self.serial_no, t))
        print(con_msg)

    def __repr__(self):
        """Simplified repr giving just the class and name."""
        return '<{}: {}>'.format(type(self).__name__, self.name)

    def _feature_detect(self):
        """
        Detect which features are available on the DAC by querying various parameters.
        """

        # Check whether EEPROM is installed
        try:
            if self._query_address(1107296256) == 21930:
                self._EEPROM_available = True
            else:
                self._EEPROM_available = False
        except DACException:
            self._EEPROM_available = False

        # Check whether we can set startup values for the DAC.
        # This requires access to the EEPROM on each slot

        # note from DV: the value never gets set to True in this driver.
        # To avoid an error of a non existing attribute, here I set it to
        # False by default
        self._VERSA_EEPROM_available = False

        try:
            # Let's temporarily pretend to be slot 0
            self._slot = 0
            self._query_address(6, versa_eeprom=True)
            del self._slot
        except DACException:
            pass

        # Check whether calibration is supported
        try:
            if self._dac_parse(self.ask("k;")):
                self._cal_supported = True
        except DACException:
            self._cal_supported = False

        # Finally try and read the DAC version and S/N. This is only possible if the EEPROM
        # is queryable.
        if self._EEPROM_available:
            self.version = self._query_address(1107296266)
            self.serial_no = self._query_address(1107296264)
        else:
            self.version = 0
            self.serial_no = 0

    def write(self, cmd):
        """
        Since all commands are echoed back, we must keep track of responses
        as well, otherwise commands receive the wrong response. Hence
        all writes must also read a response.
        """
        return self.ask(cmd)
