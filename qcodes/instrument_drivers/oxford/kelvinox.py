# OxfordInstruments_Kelvinox_IGH class, to perform the communication between the Wrapper and the device
# Copyright (c) 2017 QuTech (Delft)
# Code is available under the available under the `MIT open-source license
# <https://opensource.org/licenses/MIT>`_

# Sjaak van Diepen <c.j.vandiepen@gmail.com>, 2017
# Guenevere Prawiroatmodjo <guen@vvtp.tudelft.nl>, 2009
# Pieter de Groot <pieterdegroot@gmail.com>, 2009


from time import sleep
import visa
import logging
import numpy
from qcodes import VisaInstrument
from qcodes import validators as vals
from functools import partial


log = logging.getLogger(__name__)


class OxfordInstruments_Kelvinox_IGH(VisaInstrument):
    """
    This is the python driver for the Oxford Instruments Kelvinox IGH Dilution Refrigerator and
    Intelligent Dilution Refrigerator Power Supply (IDR PS).

    Usage:
    Initialize with
    fridge = qcodes.instrument_drivers.oxford.kelvinox.OxfordInstruments_Kelvinox_IGH(name='fridge', address='ASRL4::INSTR')

    Note: Since the ISOBUS allows for several instruments to be managed in parallel, the command
    which is sent to the device starts with '@n', where n is the ISOBUS instrument number.
    """

    def __init__(self, name, address, number=5, **kwargs):
        """
        Initializes the Oxford Instruments Kelvinox IGH Dilution Refrigerator.

        Input:
            name (string)    : name of the instrument
            address (string) : instrument address
            number (int)     : ISOBUS instrument number
        """
        log.debug('Initializing instrument')
        super().__init__(name, address, **kwargs)

        self._address = address
        self._number = number
        self._values = {}
        self.visa_handle.set_visa_attribute(visa.constants.VI_ATTR_ASRL_STOP_BITS,
                                            visa.constants.VI_ASRL_STOP_TWO)
        self._valve_map = {
            1: '9',
            2: '8',
            3: '7',
            4: '11A',
            5: '13A',
            6: '13B',
            7: '11B',
            8: '12B',
            10: '1',
            11: '5',
            12: '4',
            13: '3',
            14: '14',
            15: '10',
            16: '2',
            17: '2A',
            18: '1A',
            19: '5A',
            20: '4A'
        }

        # Add parameters
        self.add_parameter('one_K_pot_temp',
                           unit='K',
                           get_cmd=self._get_one_K_pot_temp)
        self.add_parameter('mix_chamber_temp',
                           unit='K',
                           get_cmd=self._get_mix_chamber_temp,
                           set_cmd=self._set_mix_chamber_temp)
        self.add_parameter('G1',
                           unit='mbar',
                           get_cmd=self._get_G1)
        self.add_parameter('G2',
                           unit='mbar',
                           get_cmd=self._get_G2)
        self.add_parameter('G3',
                           unit='mbar',
                           get_cmd=self._get_G3)
        self.add_parameter('P1',
                           unit='mbar',
                           get_cmd=self._get_P1)
        self.add_parameter('P2',
                           unit='mbar',
                           get_cmd=self._get_P2)
        self.add_parameter('V6_valve',
                           unit='%',
                           get_cmd=self._get_V6_valve,
                           set_cmd=self._set_V6_valve)
        self.add_parameter('V12A_valve',
                           unit='%',
                           get_cmd=self._get_V12A_valve,
                           set_cmd=self._set_V12A_valve)
        self.add_parameter('still_status',
                           get_cmd=self._get_still_status)
        self.add_parameter('sorb_status',
                           get_cmd=self._get_sorb_status)
        self.add_parameter('still_power',
                           unit='mW',
                           get_cmd=self._get_still_power,
                           set_cmd=self._set_still_power)
        self.add_parameter('sorb_temp',
                           unit='K',
                           get_cmd=self._get_sorb_temp,
                           set_cmd=self._set_sorb_temp)
        self.add_parameter('remote_status',
                           get_cmd=self._get_remote_status,
                           set_cmd=self._set_remote_status,
                           vals=vals.Ints())

        for valve in self._valve_map:
            self.add_parameter('V%s_valve' % self._valve_map[valve],
                               get_cmd=partial(
                                   self._get_valve_status, valve=valve),
                               set_cmd=partial(self._set_valve_status, valve=valve))

    def get_all(self):
        """
        Reads all implemented parameters from the instrument,
        and updates the wrapper.
        """
        log.info('reading all settings from instrument')
        self.snapshot(update=True)

    # Functions
    def _execute(self, message):
        """
        Write a command to the device

        Args:
            message (str) : write command for the device
        """
        log.info('Send the following command to the device: %s' % message)
        self.visa_handle.write('@%s%s' % (self._number, message))
        sleep(70e-3)  # wait for the device to be able to respond
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
        """Identify the device

        Returns:
             a string of the form ``'IGH    Version 3.02 (c) OXFORD 1998\\r'``

        """
        log.info('Identify the device')
        return self._execute('V')

    def remote(self):
        """Set control to remote and unlocked"""
        log.info('Set control to remote and unlocked')
        self.remote_status(3)

    def local(self):
        """Set control to local and unlocked"""
        log.info('Set control to local and unlocked')
        self.remote_status(2)

    def close(self):
        """Safely close connection"""
        log.info('Closing IPS120 connection')
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
        identity = self.identify()
        idparts = [identity[24:30], identity[:3], None, identity[15:19]]

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
        log.info('Get remote control status')
        result = self._execute('X')
        val_mapping = {0: "Local and locked",
                       1: "Remote and locked",
                       2: "Local and unlocked",
                       3: "Remote and unlocked",
                       4: "Auto-run-down",
                       5: "Auto-run-down",
                       6: "Auto-run-down",
                       7: "Auto-run-down"}
        return val_mapping[int(result[5])]

    def _set_remote_status(self, mode):
        """
        Set remote control status.

        Args:
            mode(int) :
            0 : "Local and locked",
            1 : "Remote and locked" (not available),
            2 : "Local and unlocked",
            3 : "Remote and unlocked"
        """
        status = {
            0: "Local and locked",
            2: "Local and unlocked",
            3: "Remote and unlocked",
        }
        if status.__contains__(mode):
            log.info('Setting remote control status to %s' %
                     status.get(
                         mode,
                         "Unknown"))
            self._execute('C%s' % mode)
        else:
            print('Invalid mode inserted: %s' % mode)

    def _get_one_K_pot_temp(self):
        """
        Get 1K Pot Temperature from device.

        Output:
            result (float) : 1K Pot Temperature in mK
        """
        log.info('Read 1K Pot Temperature')
        result = self._execute('R2')
        return float(result.replace('R', '')) / 1000

    def _get_mix_chamber_temp(self):
        """
        Get Mix Chamber Temperature

        Output:
            result (float) : Mix Chamber Temperature in mK
        """
        log.info('Read Mix Chamber Temperature')
        result = self._execute('R3')
        return 1e-3 * float(result.replace('R', ''))

    def set_mix_chamber_heater_mode(self, mode):
        """
        0 : off
        1 : fixed heater power
        2 : temperature control
        """
        log.info('Setting Mix Chamber Power control')
        self._execute('A%i' % mode)

    def set_mix_chamber_heater_power_range(self, mode):
        """
        1 : 2uW
        2 : 20uW
        3 : 200uW
        4 : 2mW
        5 : 20mW
        """
        log.info('Setting Mix Chamber Power range')
        self._execute('E%i' % mode)

    def _set_mix_chamber_temp(self, temperature):
        """
        Temperature in kelvin
        Between 0 and 2K
        """
        T = round(temperature / 0.1e-3)
        log.info('Setting Mix Chamber Temperature')
        self._execute('T%i' % T)

    def _get_G1(self):
        """
        Get the pressure indicated by G1

        Output:
            result (float) : G1 pressure in mbar
        """
        log.info('Read G1')
        result = self._execute('R14')
        return float(result.replace('R', '')) / 10

    def _get_G2(self):
        """
        Get the pressure indicated by G2

        Output:
            result (float) : G2 pressure in mbar
        """
        log.info('Read G2')
        result = self._execute('R15')
        return float(result.replace('R', '')) / 10

    def _get_G3(self):
        """
        Get the pressure indicated by G3

        Output:
            result (float) : G3 pressure in mbar
        """
        log.info('Read G3')
        result = self._execute('R16')
        return float(result.replace('R', '')) / 10

    def _get_P1(self):
        """
        Get the pressure indicated by P1

        Output:
            result (float) : P1 pressure in mbar
        """
        log.info('Read P1')
        result = self._execute('R20')
        return float(result.replace('R', ''))

    def _get_P2(self):
        """
        Get the pressure indicated by P2

        Output:
            result (float) : P2 pressure in mbar
        """
        log.info('Read P2')
        result = self._execute('R21')
        return float(result.replace('R', ''))

    def _get_valve_status(self, valve):
        """
        Return the status of the valve number "valve" where valve must be a number between 1 and 20 (self._map_valve)

        Output: 'On' or 'off'
        """
        result = self._execute('X')[7:15]
        # change the hexadecimal number to a 20 bit string
        full_status = numpy.binary_repr(int(result, 16), width=20)
        # reverse the order of the binary string
        full_status = full_status[::-1]
        status = full_status[valve - 1]
        val_mapping = {0: 'off', 1: 'on'}
        return val_mapping[int(status)]

    def _set_valve_status(self, status, valve):
        """
        Return the status of the valve number "valve" where valve must be a number between 1 and 20 (self._map_valve)
        status: 0 for off and 1 for on
        """
        self.remote()
        log.info('Setting valve %s status' % self._valve_map[valve])
        self._execute('P%i' % (2 * valve + numpy.mod(status + 1, 2)))
        self.local()

    def _set_V6_valve(self, status):
        """
        This set the opening of the stepper valve 6. Status should be a percentage.
        """
        self.remote()
        log.info('Setting valve 6 status')
        self._execute('G%i' % int(10 * status))
        self.local()

    def _get_V6_valve(self):
        """
        Return the opening of valve 6.

        Output:
            result(float): Opening of V6 valve in percent
        """
        result = self._execute('R7')
        return float(result.replace('R', '')) / 10

    def _set_V12A_valve(self, status):
        """
        This set the opening of the stepper valve 12. Status should be a percentage.
        """
        self.remote()
        log.info('Setting valve V12A status')
        self._execute('H%i' % int(10 * status))
        self.local()

    def _get_V12A_valve(self):
        """
        Return the opening of valve V12A.

        Output:
            result(float): Opening of valve V12A in percent
        """
        result = self._execute('R8')
        return float(result.replace('R', '')) / 10

    def rotate_Nvalve(self, value):
        """
        This set the opening of the stepper valve N. Status should be a percentage.
        """
        self.remote()
        log.info('Setting valve N status')
        value = int(value)
        if value < 1000:
            if value > -1:
                self._execute('N%i' % int(value))
                self.local()
            else:
                print('Wrong value....')
        else:
            print('Wrong value....')

    def _get_still_sorb_status(self):
        """Get the the still and sorb status
            O0 : both off
            O1 : Still ON, sorb OFF
            O2 : Still OFF, sorb on T control
            O3 : Still ON, sorb ON T control
            O4 : Still OFF, sorb on power control
            O5 : Still ON, sorb ON power control
        """
        result = self._execute('X')
        result = result[17:19]
        return result

    def _set_still_sorb_status(self, status):
        """Get the the still and sorb status
            O0 : both off
            O1 : Still ON, sorb OFF
            O2 : Still OFF, sorb on T control
            O3 : Still ON, sorb ON T control
            O4 : Still OFF, sorb on power control
            O3 : Still ON, sorb ON power control
        """
        self.remote()
        log.info('Set still and sorb status')
        self._execute(status)
        self.local()

    def _get_still_status(self):
        """ get the status of the still (on/off)"""
        status = self._get_still_sorb_status()
        if (status == 'O0') | (status == 'O2') | (status == 'O4'):
            still_status = 0
        else:
            still_status = 1
        val_mapping = {0: 'off', 1: 'on'}
        return val_mapping[int(still_status)]

    def _get_sorb_status(self):
        """ get the status of the still (on/off)"""
        status = self._get_still_sorb_status()
        if (status == 'O0') | (status == 'O1'):
            sorb_status = 0
        elif (status == 'O2') | (status == 'O3'):
            sorb_status = 1
        elif (status == 'O4') | (status == 'O5'):
            sorb_status = 2
        val_mapping = {0: 'off', 1: 'on T control', 2: 'on P control'}
        return val_mapping[sorb_status]

    def _get_still_power(self):
        """ get the power on the still"""
        log.info('Read still power')
        result = self._execute('R5')
        return float(result.replace('R', '')) / 10

    def _get_sorb_temp(self):
        """ get the temperature of the sorb"""
        log.info('Read sorb temperature')
        result = self._execute('R1')
        return float(result.replace('R', '')) / 10

    def _set_sorb_temp(self, temperature):
        """ Temperature in kelvin """
        T = round(temperature / 0.1)
        status = self._get_still_sorb_status()
        self.remote()
        if (status == 'O0'):  # turn the sorb ON
            self._set_still_sorb_status('O2')
        elif (status == 'O1'):  # turn the sorb ON
            self._set_still_sorb_status('O3')
        elif (status == 'O2') | (status == 'O3'):  # the sorb already on T control
            pass
        else:
            log.error('The sorb must be either OFF or on temperature control')

        log.info('Setting sorb temperature')
        self._execute('K%i' % T)
        self.get_sorb_status()
        self.local()

    def _set_still_power(self, temperature):
        """ power in mW"""
        P = round(temperature / 0.1)
        status = self._get_still_sorb_status()
        self.remote()
        if (status == 'O0'):  # turn the sorb ON
            self._set_still_sorb_status('O1')
        elif (status == 'O2'):  # turn the sorb ON
            self._set_still_sorb_status('O3')
        elif (status == 'O4'):  # the sorb already on T control
            self._set_still_sorb_status('O5')
        else:
            pass

        log.info('Setting still power')
        self._execute('S%i' % P)
        self._get_still_status()
        self.local()
