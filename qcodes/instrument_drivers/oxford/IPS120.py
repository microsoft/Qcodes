# OxfordInstruments_IPS120.py class, to perform the communication between the Wrapper and the device
# Copyright (c) 2017 QuTech (Delft)
# Code is available under the available under the `MIT open-source license <https://opensource.org/licenses/MIT>`_

# Sjaak van Diepen <c.j.vandiepen@gmail.com>, 2017
# Takafumi Fujita <t.fujita@tudelft.nl>, 2016
# Mohammad Shafiei <m.shafiei@tudelft.nl>, 2011
# Guenevere Prawiroatmodjo <guen@vvtp.tudelft.nl>, 2009
# Pieter de Groot <pieterdegroot@gmail.com>, 2009


import logging
from qcodes import VisaInstrument
from qcodes import validators as vals
from time import sleep
import visa


log = logging.getLogger(__name__)

class OxfordInstruments_IPS120(VisaInstrument):
    """This is the driver for the Oxford Instruments IPS 120 Magnet Power Supply

    The IPS 120 can connect through both RS232 serial as well as GPIB. The
    commands sent in both cases are similar. When using the serial connection,
    commands are prefaced with '@n' where n is the ISOBUS number.
    """

    _GET_STATUS_MODE = {
            0: "Amps, Magnet sweep: fast",
            1: "Tesla, Magnet sweep: fast",
            4: "Amps, Magnet sweep: slow",
            5: "Tesla, Magnet sweep: slow",
            8 : "Amps, (Magnet sweep: unaffected)",
            9 : "Tesla, (Magnet sweep: unaffected)"}

    _GET_STATUS_MODE2 = {
            0: "At rest",
            1: "Sweeping",
            2: "Sweep limiting",
            3: "Sweeping & sweep limiting",
            5: "Unknown"}

    _GET_STATUS_SWITCH_HEATER = {
            0: "Off magnet at zero (switch closed)",
            1: "On (switch open)",
            2: "Off magnet at field (switch closed)",
            5: "Heater fault (heater is on but current is low)",
            8: "No switch fitted"}

    _GET_STATUS_REMOTE = {
            0: "Local and locked",
            1: "Remote and locked",
            2: "Local and unlocked",
            3: "Remote and unlocked",
            4: "Auto-run-down",
            5: "Auto-run-down",
            6: "Auto-run-down",
            7: "Auto-run-down"}

    _GET_SYSTEM_STATUS = {
            0: "Normal",
            1: "Quenched",
            2: "Over Heated",
            3: "Warming Up",
            4: "Fault"}

    _GET_SYSTEM_STATUS2 = {
            0: "Normal",
            1: "On positive voltage limit",
            2: "On negative voltage limit",
            3: "Outside negative current limit",
            4: "Outside positive current limit"}

    _GET_POLARITY_STATUS1 = {
            0: "Desired: Positive, Magnet: Positive, Commanded: Positive",
            1: "Desired: Positive, Magnet: Positive, Commanded: Negative",
            2: "Desired: Positive, Magnet: Negative, Commanded: Positive",
            3: "Desired: Positive, Magnet: Negative, Commanded: Negative",
            4: "Desired: Negative, Magnet: Positive, Commanded: Positive",
            5: "Desired: Negative, Magnet: Positive, Commanded: Negative",
            6: "Desired: Negative, Magnet: Negative, Commanded: Positive",
            7: "Desired: Negative, Magnet: Negative, Commanded: Negative"}

    _GET_POLARITY_STATUS2 = {
            1: "Negative contactor closed",
            2: "Positive contactor closed",
            3: "Both contactors open",
            4: "Both contactors closed"}

    _SET_ACTIVITY = {
            0: "Hold",
            1: "To set point",
            2: "To zero"}

    _WRITE_WAIT = 100e-3 # seconds

    def __init__(self, name, address, use_gpib=False, number=2, **kwargs):
        """Initializes the Oxford Instruments IPS 120 Magnet Power Supply.

        Args:
            name (str)    : name of the instrument
            address (str) : instrument address
            use_gpib (bool)  : whether to use GPIB or serial
            number (int)     : ISOBUS instrument number. Ignored if using GPIB.
        """
        super().__init__(name, address, terminator='\r', **kwargs)

        self._address = address
        self._number = number
        self._values = {}
        self._use_gpib = use_gpib

        # Add parameters
        self.add_parameter('mode',
                           get_cmd=self._get_mode,
                           set_cmd=self._set_mode,
                           vals=vals.Ints())
        self.add_parameter('mode2',
                           get_cmd=self._get_mode2)
        self.add_parameter('activity',
                           get_cmd=self._get_activity,
                           set_cmd=self._set_activity,
                           vals=vals.Ints())
        self.add_parameter('switch_heater',
                           get_cmd=self._get_switch_heater,
                           set_cmd=self._set_switch_heater,
                           vals=vals.Ints())
        self.add_parameter('field_setpoint',
                           unit='T',
                           get_cmd=self._get_field_setpoint,
                           set_cmd=self._set_field_setpoint,
                           vals=vals.Numbers(-14, 14))
        self.add_parameter('sweeprate_field',
                           unit='T/min',
                           get_cmd=self._get_sweeprate_field,
                           set_cmd=self._set_sweeprate_field,
                           vals=vals.Numbers(0, 0.7))
        self.add_parameter('system_status',
                           get_cmd=self._get_system_status)
        self.add_parameter('system_status2',
                           get_cmd=self._get_system_status2)
        self.add_parameter('polarity',
                           get_cmd=self._get_polarity)
        self.add_parameter('voltage',
                           unit='V',
                           get_cmd=self._get_voltage)
        self.add_parameter('voltage_limit',
                           unit='V',
                           get_cmd=self._get_voltage_limit)

        # Find the F field limits
        MaxField = self.field_setpoint.vals._max_value
        MinField = self.field_setpoint.vals._min_value
        MaxFieldSweep = self.sweeprate_field.vals._max_value
        MinFieldSweep = self.sweeprate_field.vals._min_value
        # A to B conversion
        ABconversion = 115.733 / 14  # Ampere per Tesla
        self.add_parameter('current_setpoint',
                           unit='A',
                           get_cmd=self._get_current_setpoint,
                           set_cmd=self._set_current_setpoint,
                           vals=vals.Numbers(ABconversion * MinField,
                                             ABconversion * MaxField))
        self.add_parameter('sweeprate_current',
                           unit='A/min',
                           get_cmd=self._get_sweeprate_current,
                           set_cmd=self._set_sweeprate_current,
                           vals=vals.Numbers(ABconversion * MinFieldSweep,
                                             ABconversion * MaxFieldSweep))
        self.add_parameter('remote_status',
                           get_cmd=self._get_remote_status,
                           set_cmd=self._set_remote_status,
                           vals=vals.Ints())
        self.add_parameter('current',
                           unit='A',
                           get_cmd=self._get_current)
        self.add_parameter('magnet_current',
                           unit='A',
                           get_cmd=self._get_magnet_current)
        self.add_parameter('field',
                           unit='T',
                           get_cmd=self._get_field)
        self.add_parameter('persistent_current',
                           unit='A',
                           get_cmd=self._get_persistent_current)
        self.add_parameter('persistent_field',
                           unit='T',
                           get_cmd=self._get_persistent_field)
        self.add_parameter('magnet_inductance',
                           unit='H',
                           get_cmd=self._get_magnet_inductance)
        self.add_parameter('lead_resistance',
                           unit='mOhm',
                           get_cmd=self._get_lead_resistance)
        self.add_parameter('current_limit_lower',
                           unit='A',
                           get_cmd=self._get_current_limit_lower)
        self.add_parameter('current_limit_upper',
                           unit='A',
                           get_cmd=self._get_current_limit_upper)
        self.add_parameter('heater_current',
                           unit='mA',
                           get_cmd=self._get_heater_current)
        self.add_parameter('trip_field',
                           unit='T',
                           get_cmd=self._get_trip_field)
        self.add_parameter('trip_current',
                           unit='A',
                           get_cmd=self._get_trip_current)

        if not self._use_gpib:
            self.visa_handle.set_visa_attribute(
                    visa.constants.VI_ATTR_ASRL_STOP_BITS,
                    visa.constants.VI_ASRL_STOP_TWO)
            # to handle VisaIOError which occurs at first read
            try:
                self.visa_handle.write('@%s%s' % (self._number, 'V'))
                sleep(self._WRITE_WAIT)
                self._read()
            except visa.VisaIOError:
                pass

    def get_all(self):
        """
        Reads all implemented parameters from the instrument,
        and updates the wrapper.
        """
        self.snapshot(update=True)

    def _execute(self, message):
        """
        Write a command to the device and return the result.

        Args:
            message (str) : write command for the device

        Returns:
            Response from the device as a string.
        """
        self.log.info('Send the following command to the device: %s' % message)

        if self._use_gpib:
            return self.ask(message)

        self.visa_handle.write('@%s%s' % (self._number, message))
        sleep(self._WRITE_WAIT)  # wait for the device to be able to respond
        result = self._read()
        if result.find('?') >= 0:
            print("Error: Command %s not recognized" % message)
        else:
            return result

    def _read(self):
        """
        Reads the total bytes in the buffer and outputs as a string.

        Returns:
            message (str)
        """
        bytes_in_buffer = self.visa_handle.bytes_in_buffer
        with(self.visa_handle.ignore_warning(visa.constants.VI_SUCCESS_MAX_CNT)):
            mes = self.visa_handle.visalib.read(
                self.visa_handle.session, bytes_in_buffer)
        mes = str(mes[0].decode())
        return mes

    def identify(self):
        """Identify the device"""
        self.log.info('Identify the device')
        return self._execute('V')

    def examine(self):
        """Examine the status of the device"""
        self.log.info('Examine status')

        print('System Status: ')
        print(self.system_status())

        print('Activity: ')
        print(self.activity())

        print('Local/Remote status: ')
        print(self.remote_status())

        print('Switch heater: ')
        print(self.switch_heater())

        print('Mode: ')
        print(self.mode())

        print('Polarity: ')
        print(self.polarity())

    def remote(self):
        """Set control to remote and unlocked"""
        self.log.info('Set control to remote and unlocked')
        self.remote_status(3)

    def local(self):
        """Set control to local and unlocked"""
        self.log.info('Set control to local and unlocked')
        self.remote_status(2)

    def close(self):
        """Safely close connection"""
        self.log.info('Closing IPS120 connection')
        self.local()
        super().close()

    def get_idn(self):
        """
        Overides the function of Instrument since IPS120 does not support `*IDN?`

        This string is supposed to be a comma-separated list of vendor, model,
        serial, and firmware, but semicolon and colon are also common
        separators so we accept them here as well.

        Returns:
            A dict containing vendor, model, serial, and firmware.
        """
        idparts = ['Oxford Instruments', 'IPS120', None, None]

        return dict(zip(('vendor', 'model', 'serial', 'firmware'), idparts))

    def _get_remote_status(self):
        """
        Get remote control status

        Returns:
            result(str) :
            "Local & locked",
            "Remote & locked",
            "Local & unlocked",
            "Remote & unlocked",
            "Auto-run-down",
            "Auto-run-down",
            "Auto-run-down",
            "Auto-run-down"
        """
        self.log.info('Get remote control status')
        result = self._execute('X')
        return self._GET_STATUS_REMOTE[int(result[6])]

    def _set_remote_status(self, mode):
        """
        Set remote control status.

        Args:
            mode(int): Refer to _GET_STATUS_REMOTE for allowed values and
            meanings.
        """
        if mode in self._GET_STATUS_REMOTE.keys():
            self.log.info('Setting remote control status to %s'
                    % self._GET_STATUS_REMOTE[mode])
            self._execute('C%s' % mode)
        else:
            print('Invalid mode inserted: %s' % mode)

    def _get_system_status(self):
        """
        Get the system status

        Returns:
            result (str) :
            "Normal",
            "Quenched",
            "Over Heated",
            "Warming Up",
            "Fault"
        """
        result = self._execute('X')
        self.log.info('Getting system status')
        return self._GET_SYSTEM_STATUS[int(result[1])]

    def _get_system_status2(self):
        """
        Get the system status

        Returns:
            result (str) :
            "Normal",
            "On positive voltage limit",
            "On negative voltage limit",
            "Outside negative current limit",
            "Outside positive current limit"
        """
        result = self._execute('X')
        self.log.info('Getting system status')
        return self._GET_SYSTEM_STATUS2[int(result[2])]

    def _get_current(self):
        """
        Demand output current of device

        Returns:
            result (float) : output current in Amp
        """
        self.log.info('Read output current')
        result = self._execute('R0')
        return float(result.replace('R', ''))

    def _get_voltage(self):
        """
        Demand measured output voltage of device

        Returns:
            result (float) : output voltage in Volt
        """
        self.log.info('Read output voltage')
        result = self._execute('R1')
        return float(result.replace('R', ''))

    def _get_magnet_current(self):
        """
        Demand measured magnet current of device

        Returns:
            result (float) : measured magnet current in Amp
        """
        self.log.info('Read measured magnet current')
        result = self._execute('R2')
        return float(result.replace('R', ''))

    def _get_current_setpoint(self):
        """
        Return the set point (target current)

        Returns:
            result (float) : Target current in Amp
        """
        self.log.info('Read set point (target current)')
        result = self._execute('R5')
        return float(result.replace('R', ''))

    def _set_current_setpoint(self, current):
        """
        Set current setpoint (target current)

        Args:
            current (float) : target current in Amp
        """
        self.log.info('Setting target current to %s' % current)
        self.remote()
        self._execute('I%s' % current)
        self.local()
        self.field_setpoint()

    def _get_sweeprate_current(self):
        """
        Return sweep rate (current)

        Returns:
            result (float) : sweep rate in A/min
        """
        self.log.info('Read sweep rate (current)')
        result = self._execute('R6')
        return float(result.replace('R', ''))

    def _set_sweeprate_current(self, sweeprate):
        """
        Set sweep rate (current)

        Args:
            sweeprate(float) : Sweep rate in A/min.
        """
        self.remote()
        self.log.info('Set sweep rate (current) to %s A/min' % sweeprate)
        self._execute('S%s' % sweeprate)
        self.local()
        self.sweeprate_field()

    def _get_field(self):
        """
        Demand output field

        Returns:
            result (float) : magnetic field in Tesla
        """
        self.log.info('Read output field')
        result = self._execute('R7')
        return float(result.replace('R', ''))

    def _get_field_setpoint(self):
        """
        Return the set point (target field)

        Returns:
            result (float) : Field set point in Tesla
        """
        self.log.info('Read field set point')
        result = self._execute('R8')
        return float(result.replace('R', ''))

    def _set_field_setpoint(self, field):
        """
        Set the field set point (target field)

        Args:
            field (float) : target field in Tesla
        """
        self.log.info('Setting target field to %s' % field)
        self.remote()
        self._execute('J%s' % field)
        self.local()
        self.current_setpoint()

    def _get_sweeprate_field(self):
        """
        Return sweep rate (field)

        Returns:
            result (float) : sweep rate in Tesla/min
        """
        self.log.info('Read sweep rate (field)')
        result = self._execute('R9')
        return float(result.replace('R', ''))

    def _set_sweeprate_field(self, sweeprate):
        """
        Set sweep rate (field)

        Args:
            sweeprate(float) : Sweep rate in Tesla/min.
        """
        self.log.info('Set sweep rate (field) to %s Tesla/min' % sweeprate)
        self.remote()
        self._execute('T%s' % sweeprate)
        self.local()
        self.sweeprate_current()

    def _get_voltage_limit(self):
        """
        Return voltage limit

        Returns:
            result (float) : voltage limit in Volt
        """
        self.log.info('Read voltage limit')
        result = self._execute('R15')
        result = float(result.replace('R', ''))
        self.voltage.vals = vals.Numbers(-result, result)
        return result

    def _get_persistent_current(self):
        """
        Return persistent magnet current

        Returns:
            result (float) : persistent magnet current in Amp
        """
        self.log.info('Read persistent magnet current')
        result = self._execute('R16')
        return float(result.replace('R', ''))

    def _get_trip_current(self):
        """
        Return trip current

        Returns:
            result (float) : trip current om Amp
        """
        self.log.info('Read trip current')
        result = self._execute('R17')
        return float(result.replace('R', ''))

    def _get_persistent_field(self):
        """
        Return persistent magnet field

        Returns:
            result (float) : persistent magnet field in Tesla
        """
        self.log.info('Read persistent magnet field')
        result = self._execute('R18')
        return float(result.replace('R', ''))

    def _get_trip_field(self):
        """
        Return trip field

        Returns:
            result (float) : trip field in Tesla
        """
        self.log.info('Read trip field')
        result = self._execute('R19')
        return float(result.replace('R', ''))

    def _get_heater_current(self):
        """
        Return switch heater current

        Returns:
            result (float) : switch heater current in milliAmp
        """
        self.log.info('Read switch heater current')
        result = self._execute('R20')
        return float(result.replace('R', ''))

    def _get_current_limit_upper(self):
        """
        Return safe current limit, most positive

        Returns:
            result (float) : safe current limit, most positive in Amp
        """
        self.log.info('Read safe current limit, most positive')
        result = self._execute('R22')
        return float(result.replace('R', ''))

    def _get_current_limit_lower(self):
        """
        Return safe current limit, most negative

        Returns:
            result (float) : safe current limit, most negative in Amp
        """
        self.log.info('Read safe current limit, most negative')
        result = self._execute('R21')
        return float(result.replace('R', ''))

    def _get_lead_resistance(self):
        """
        Return lead resistance

        Returns:
            result (float) : lead resistance in milliOhm
        """
        self.log.info('Read lead resistance')
        result = self._execute('R23')
        return float(result.replace('R', ''))

    def _get_magnet_inductance(self):
        """
        Return magnet inductance

        Returns:
            result (float) : magnet inductance in Henry
        """
        self.log.info('Read magnet inductance')
        result = self._execute('R24')
        return float(result.replace('R', ''))

    def _get_activity(self):
        """
        Get the activity of the magnet. Possibilities: Hold, Set point, Zero or Clamp.

        Returns:
            result(str) : "Hold", "Set point", "Zero" or "Clamp".
        """
        self.log.info('Get activity of the magnet.')
        result = self._execute('X')
        return status[int(result[4])]

    def _set_activity(self, mode):
        """
        Set the activity to Hold, To Set point or To Zero.

        Args:
            mode (int): See _SET_ACTIVITY for values and meanings.
        """
        if mode in self._SET_ACTIVITY.keys():
            self.log.info('Setting magnet activity to %s'
                    % self._SET_ACTIVITY[mode])
            self.remote()
            self._execute('A%s' % mode)
            self.local()
        else:
            print('Invalid mode inserted.')

    def hold(self):
        """Set the device activity to Hold"""
        self.activity(0)

    def to_setpoint(self):
        """Set the device activity to "To set point". This initiates a sweep."""
        self.activity(1)

    def to_zero(self):
        """
        Set the device activity to "To zero". This sweeps te magnet back to zero.
        """
        self.activity(2)

    def _get_switch_heater(self):
        """
        Get the switch heater status.

        Returns:
            result(str): See _GET_STATUS_SWITCH_HEATER.
        """
        self.log.info('Get switch heater status')
        result = self._execute('X')
        return self._GET_STATUS_SWITCH_HEATER[int(result[8])]

    def _set_switch_heater(self, mode):
        """
        Set the switch heater Off or On. Note: After issuing a command it is necessary to wait
        several seconds for the switch to respond.
        Args:
            mode (int) :
            0 : Off
            1 : On
        """
        if mode in [0, 1]:
            self.log.info('Setting switch heater to %d' % mode)
            self.remote()
            self._execute('H%s' % mode)
            print("Setting switch heater... (wait 40s)")
            self.local()
            sleep(40)
        else:
            print('Invalid mode inserted.')
        sleep(0.1)
        self.switch_heater()

    def heater_on(self):
        """Switch the heater on, with PSU = Magnet current check"""
        current_in_magnet = self.persistent_current()
        current_in_leads = self.current()
        if self.switch_heater() == self._GET_STATUS_SWITCH_HEATER[1]:
            print('Heater is already on!')
        else:
            if self.mode2() == self._GET_STATUS_MODE2[0]:
                if current_in_leads == current_in_magnet:
                    self.switch_heater(1)
                else:
                    print('Current in the leads is not matching persistent current!')
            else:
                print('Magnet supply not at rest, cannot switch on heater!')
        self.switch_heater()

    def set_persistent(self):
        """
        Puts magnet into persistent mode

        Note: After turning of the switch heater we will wait for additional 20
        seconds before we put the current to zero. This is done to make sure
        that the switch heater is cold enough and becomes superconducting.
        """
        if self.mode2() == self._GET_STATUS_MODE2[0]:
            self.heater_off()
            print('Waiting for the switch heater to become superconducting')
            sleep(20)
            self.to_zero()
            self.get_all()
        else:
            print('Magnet is not at rest, cannot put it in persistent mode')
        self.get_all()

    def leave_persistent_mode(self):
        """
        Read out persistent current, match the current in the leads to that current
        and switch on heater
        """
        if self.switch_heater() == self._GET_STATUS_SWITCH_HEATER[2]:
            field_in_magnet = self.persistent_field()
            field_in_leads = self.field()
            self.hold()
            self.field_setpoint(field_in_magnet)
            self.to_setpoint()

            while field_in_leads != field_in_magnet:
                field_in_leads = self.field()
            self.heater_on()
            self.hold()

        elif self.switch_heater() == self._GET_STATUS_SWITCH_HEATER[1]:
            print('Heater is already on, so the magnet was not in persistent mode')
        elif self.switch_heater() == self._GET_STATUS_SWITCH_HEATER[0]:
            print('Heater is off, but magnet is not in persistent mode. Please, check magnet locally!')

        self.get_all()

    def run_to_field(self, field_value):
        """
        Go to field value

        Args:
            field_value (float): the magnetic field value to go to in Tesla
        """

        if self.switch_heater() == self._GET_STATUS_SWITCH_HEATER[1]:
            self.hold()
            self.field_setpoint(field_value)
            self.to_setpoint()
        else:
            print('Switch heater is off, cannot change the field.')
        self.get_all()

    def run_to_field_wait(self, field_value):
        """
        Go to field value and wait until it's done sweeping.

        Args:
            field_value (float): the magnetic field value to go to in Tesla
        """
        if self.switch_heater() == self._GET_STATUS_SWITCH_HEATER[1]:
            self.hold()
            self.field_setpoint(field_value)
            self.remote()
            self.to_setpoint()
            magnet_mode = self.mode2()
            while magnet_mode != self._GET_STATUS_MODE2[0]:
                magnet_mode = self.mode2()
                sleep(0.5)
        else:
            print('Switch heater is off, cannot change the field.')
        self.get_all()
        self.local()

    def heater_off(self):
        """Switch the heater off"""
        if (self.switch_heater() == self._GET_STATUS_SWITCH_HEATER[0] or
                self.switch_heater() == self._GET_STATUS_SWITCH_HEATER[2]):
            print('Heater is already off!')
        else:
            if self.mode2() == self._GET_STATUS_MODE2[0]:
                self.switch_heater(0)
            else:
                print('Magnet is not at rest, cannot switch of the heater!')

    def _get_mode(self):
        """
        Get the mode of the device

        Returns:
            mode(str): See _GET_STATUS_MODE.
        """
        self.log.info('Get device mode')
        result = self._execute('X')
        return self._GET_STATUS_MODE[int(result[10])]

    def _get_mode2(self):
        """
        Get the sweeping mode of the device

        Returns:
            mode(str): See _GET_STATUS_MODE2.
        """
        self.log.info('Get device mode')
        result = self._execute('X')
        return self._GET_STATUS_MODE2[int(result[11])]

    def _set_mode(self, mode):
        """
        Args:
            mode(int): Refer to _GET_STATUS_MODE dictionary for the allowed
            mode values and meanings.
        """
        if mode in self._GET_STATUS_MODE.keys():
            self.log.info('Setting device mode to %s' % self._GET_STATUS_MODE[mode])
            self.remote()
            self._execute('M%s' % mode)
            self.local()
        else:
            print('Invalid mode inserted.')

    def _get_polarity(self):
        """
        Get the polarity of the output current

        Returns:
            result (str): See _GET_POLARITY_STATUS1 and _GET_POLARITY_STATUS2.
        """
        self.log.info('Get device polarity')
        result = self._execute('X')
        return self._GET_POLARITY_STATUS1[int(result[13])] + \
            ", " + self._GET_POLARITY_STATUS2[int(result[14])]
