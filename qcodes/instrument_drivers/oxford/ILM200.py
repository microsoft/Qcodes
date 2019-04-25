# OxfordInstruments_ILM200.py class, to perform the communication between the Wrapper and the device
# Copyright (c) 2017 QuTech (Delft)
# Code is available under the available under the `MIT open-source license <https://opensource.org/licenses/MIT>`__
#
# Pieter Eendebak <pieter.eendebak@tno.nl>, 2017
# Takafumi Fujita <t.fujita@tudelft.nl>, 2016
# Guenevere Prawiroatmodjo <guen@vvtp.tudelft.nl>, 2009
# Pieter de Groot <pieterdegroot@gmail.com>, 2009


from time import sleep
import visa
import logging
from qcodes import VisaInstrument


class OxfordInstruments_ILM200(VisaInstrument):
    """
    This is the qcodes driver for the Oxford Instruments ILM 200 Helium Level Meter.

    Usage:
    Initialize with
    <name> = instruments.create('name', 'OxfordInstruments_ILM200', address='<Instrument address>')
    <Instrument address> = ASRL4::INSTR

    Note: Since the ISOBUS allows for several instruments to be managed in parallel, the command
    which is sent to the device starts with '@n', where n is the ISOBUS instrument number.

    """

    def __init__(self, name, address, number=1, **kwargs):
        """
        Initializes the Oxford Instruments ILM 200 Helium Level Meter.

        Args:
            name (str): name of the instrument
            address (str): instrument address
            number (int): ISOBUS instrument number (number=1 is specific to the ILM in F008)

        Returns:
            None
        """
        logging.debug(__name__ + ' : Initializing instrument')
        super().__init__(name, address, **kwargs)

        self.visa_handle.set_visa_attribute(visa.constants.VI_ATTR_ASRL_STOP_BITS,
                                            visa.constants.VI_ASRL_STOP_TWO)
        self._address = address
        self._number = number
        self._values = {}

        self.add_parameter('level',
                           label='level',
                           get_cmd=self._do_get_level,
                           unit='%')
        self.add_parameter('status',
                           get_cmd=self._do_get_status)
        self.add_parameter('rate',
                           get_cmd=self._do_get_rate,
                           set_cmd=self._do_set_rate)

        # a dummy command to avoid the initial error
        try:
            self.get_idn()
            sleep(70e-3)  # wait for the device to be able to respond
            self._read()  # to flush the buffer
        except Exception as ex:
            logging.debug(ex)

    def _execute(self, message):
        """
        Write a command to the device and read answer. This function writes to
        the buffer by adding the device number at the front, instead of 'ask'.

        Args:
            message (str) : write command for the device

        Returns:
            None
        """
        logging.info(
            __name__ + ' : Send the following command to the device: %s' % message)
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

        Args:
            None

        Returns:
            message (str)
        """
        # because protocol has no termination chars the read reads the number
        # of bytes in the buffer
        bytes_in_buffer = self.visa_handle.bytes_in_buffer
        # a workaround for a timeout error in the pyvsia read_raw() function
        with(self.visa_handle.ignore_warning(visa.constants.VI_SUCCESS_MAX_CNT)):
            mes = self.visa_handle.visalib.read(
                self.visa_handle.session, bytes_in_buffer)
        # cannot be done on same line for some reason
        mes = str(mes[0].decode())
        return mes

    def get_idn(self):
        """
        Overrides the function of Instrument since ILM does not support `*IDN?`

        This string is supposed to be a
        comma-separated list of vendor, model, serial, and firmware, but
        semicolon and colon are also common separators so we accept them here
        as well.

        Returns:
            A dict containing vendor, model, serial, and firmware.
        """
        try:
            idstr = ''  # in case self.ask fails
            idstr = self._get_version().split()
            # form is supposed to be comma-separated, but we've seen
            # other separators occasionally
            idparts = [idstr[3] + ' ' + idstr[4], idstr[0], idstr[5],
                       idstr[1] + ' ' + idstr[2]]
            # in case parts at the end are missing, fill in None
            if len(idparts) < 4:
                idparts += [None] * (4 - len(idparts))
        except Exception as ex:
            logging.warn('Error getting or interpreting *IDN?: ' + repr(idstr))
            logging.debug(ex)
            idparts = [None, None, None, None]

        return dict(zip(('vendor', 'model', 'serial', 'firmware'), idparts))

    def get_all(self):
        """
        Reads all implemented parameters from the instrument,
        and updates the wrapper.
        """
        logging.info(__name__ + ' : reading all settings from instrument')
        self.level.get()
        self.status.get()
        self.rate.get()

    def close(self):
        """
        Safely close connection
        """
        logging.info(__name__ + ' : Closing ILM200 connection')
        self.local()
        super().close()

    # Functions: Monitor commands
    def _get_version(self):
        """
        Identify the device

        Args:
            None

        Returns:
            identification (str): should be 'ILM200 Version 1.08 (c) OXFORD 1994\r'
        """
        logging.info(__name__ + ' : Identify the device')
        return self._execute('V')

    def _do_get_level(self):
        """
        Get Helium level of channel 1.

        Args:
            None

        Returns:
            result (float) : Helium level
        """
        logging.info(__name__ + ' : Read level of channel 1')
        result = self._execute('R1')
        return float(result.replace("R", "")) / 10

    def _do_get_status(self):
        """
        Get status of the device.
        """
        logging.info(__name__ + ' : Get status of the device.')
        result = self._execute('X')
        usage = {
            0: "Channel not in use",
            1: "Channel used for Nitrogen level",
            2: "Channel used for Helium Level (Normal pulsed operation)",
            3: "Channel used for Helium Level (Continuous measurement)",
            9: "Error on channel (Usually means probe unplugged)"
        }
        # current_flowing = {
        # 0 : "Curent not flowing in Helium Probe Wire",
        # 1 : "Curent not flowing in Helium Probe Wire"
        # }
        # auto_fill_status = {
        # 00 : "End Fill (Level > FULL)",
        # 01 : "Not Filling (Level < FULL, Level > FILL)",
        # 10 : "Filling (Level < FULL, Level > FILL)",
        # 11 : "Start Filling (Level < FILL)"
        # }
        return usage.get(int(result[1]), "Unknown")

    def _do_get_rate(self):
        """
        Get helium meter channel 1 probe rate

        Input:
            None

        Output:
            rate(int) :
            0 : "SLOW"
            1 : "FAST"
        """
        rate = {
            1: "1 : Helium Probe in FAST rate",
            0: "0 : Helium Probe in SLOW rate"
        }
        result = self._execute('X')
        return rate.get(int(format(int(result[5:7]), '08b')[6]), "Unknown")

    def remote(self):
        """
        Set control to remote & locked
        """
        logging.info(__name__ + ' : Set control to remote & locked')
        self.set_remote_status(1)

    def local(self):
        """
        Set control to local & locked
        """
        logging.info(__name__ + ' : Set control to local & locked')
        self.set_remote_status(0)

    def set_remote_status(self, mode):
        """
        Set remote control status.

        Args:
            mode(int) :
            0 : "Local and locked",
            1 : "Remote and locked",
            2 : "Local and unlocked",
            3 : "Remote and unlocked",

        Returns:
            None
        """
        status = {
            0: "Local and locked",
            1: "Remote and locked",
            2: "Local and unlocked",
            3: "Remote and unlocked",
        }
        logging.info(__name__ + ' : Setting remote control status to %s' %
                     status.get(mode, "Unknown"))
        self._execute('C%s' % mode)

    # Functions: Control commands (only recognised when in REMOTE control)
    def set_to_slow(self):
        """
        Set helium meter channel 1 to slow mode.
        """
        self.set_remote_status(1)
        logging.info(__name__ + ' : Setting Helium Probe in SLOW rate')
        self._execute('S1')
        self.set_remote_status(3)

    def set_to_fast(self):
        """
        Set helium meter channel 1 to fast mode.
        """
        self.set_remote_status(1)
        logging.info(__name__ + ' : Setting Helium Probe in FAST rate')
        self._execute('T1')
        self.set_remote_status(3)

    def _do_set_rate(self, rate):
        """
        Set helium meter channel 1 probe rate

        Args:
            rate(int) :
            0 : "SLOW"
            1 : "FAST"
        """
        self.set_remote_status(1)
        if rate == 0:
            self.set_to_slow()
        elif rate == 1:
            self.set_to_fast()
        self.set_remote_status(3)
        logging.info(self._do_get_rate())
