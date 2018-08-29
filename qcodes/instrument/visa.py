"""Visa instrument driver based on pyvisa."""
from typing import Sequence
import warnings
import logging

import visa
import pyvisa.constants as vi_const
import pyvisa.resources

from .base import Instrument
import qcodes.utils.validators as vals

log = logging.getLogger(__name__)


class VisaInstrument(Instrument):

    """
    Base class for all instruments using visa connections.

    Args:
        name (str): What this instrument is called locally.

        address (str): The visa resource name to use to connect.
            Optionally includes '@<backend>' at the end. For example,
            'ASRL2' will open COM2 with the default NI backend, but
            'ASRL2@py' will open COM2 using pyvisa-py. Note that qcodes
            does not install (or even require) ANY backends, it is up to
            the user to do that. see eg:
            http://pyvisa.readthedocs.org/en/stable/names.html

        timeout (number): seconds to allow for responses. Default 5.

        terminator: Read termination character(s) to look for. Default ''.

        device_clear: Perform a device clear. Default True.

        metadata (Optional[Dict]): additional static metadata to add to this
            instrument's JSON snapshot.

    See help for ``qcodes.Instrument`` for additional information on writing
    instrument subclasses.

    Attributes:
        visa_handle (pyvisa.resources.Resource): The communication channel.
    """

    def __init__(self, name, address=None, timeout=5,
                 terminator='', device_clear=True, visalib=None, **kwargs):

        super().__init__(name, **kwargs)

        self.add_parameter('timeout',
                           get_cmd=self._get_visa_timeout,
                           set_cmd=self._set_visa_timeout,
                           unit='s',
                           vals=vals.MultiType(vals.Numbers(min_value=0),
                                               vals.Enum(None)))

        # backwards-compatibility
        if address and '@' in address:
            address, visa_library = address.split('@')
            if visalib:
                warnings.warn('You have specified the VISA library in two '
                              'different ways. Please do not include "@" in '
                              'the address kwarg and only use the visalib '
                              'kwarg for that.')
                self.visalib = visalib
            else:
                warnings.warn('You have specified the VISA library using '
                              'an "@" in the address kwarg. Please use the '
                              'visalib kwarg instead.')
                self.visalib = '@' + visa_library
        else:
            self.visalib = visalib

        self.visabackend = None

        try:
            self.set_address(address)
        except Exception as e:
            log.info(f"Could not connect to {name} instrument at {address}")
            self.close()
            raise e

        if device_clear:
            self.device_clear()

        self.set_terminator(terminator)
        self.timeout.set(timeout)

    def set_address(self, address):
        """
        Set the address for this instrument.

        Args:
            address: The visa resource name to use to connect. The address
                should be the actual address and just that. If you wish to
                change the backend for VISA, use the self.visalib attribute
                (and then call this function).
        """

        # in case we're changing the address - close the old handle first
        if getattr(self, 'visa_handle', None):
            self.visa_handle.close()

        if self.visalib:
            log.info('Opening PyVISA Resource Manager with visalib:'
                     ' {}'.format(self.visalib))
            resource_manager = visa.ResourceManager(self.visalib)
            self.visabackend = self.visalib.split('@')[1]
        else:
            log.info('Opening PyVISA Resource Manager with default'
                     ' backend.')
            resource_manager = visa.ResourceManager()
            self.visabackend = 'ni'

        log.info('Opening PyVISA resource at address: {}'.format(address))
        self.visa_handle = resource_manager.open_resource(address)
        self._address = address

    def device_clear(self):
        """Clear the buffers of the device"""

        # Serial instruments have a separate flush method to clear
        # their buffers which behaves differently to clear. This is
        # particularly important for instruments which do not support
        # SCPI commands.

        # Simulated instruments do not support a handle clear
        if self.visabackend == 'sim':
            return

        if isinstance(self.visa_handle, pyvisa.resources.SerialInstrument):
            self.visa_handle.flush(
                vi_const.VI_READ_BUF_DISCARD | vi_const.VI_WRITE_BUF_DISCARD)
        else:
            status_code = self.visa_handle.clear()
            if status_code is not None:
                log.warning("Cleared visa buffer on "
                            "{} with status code {}".format(self.name,
                                                            status_code))

    def set_terminator(self, terminator):
        r"""
        Change the read terminator to use.

        Args:
            terminator (str): Character(s) to look for at the end of a read.
                eg. '\r\n'.
        """
        self.visa_handle.write_termination = terminator
        self.visa_handle.read_termination = terminator
        self._terminator = terminator

        if self.visabackend == 'sim':
                self.visa_handle.write_termination = terminator

    def _set_visa_timeout(self, timeout):

        if timeout is None:
            self.visa_handle.timeout = None
        else:
            # pyvisa uses milliseconds but we use seconds
            self.visa_handle.timeout = timeout * 1000.0

    def _get_visa_timeout(self):

        timeout_ms = self.visa_handle.timeout
        if timeout_ms is None:
            return None
        else:
            # pyvisa uses milliseconds but we use seconds
            return timeout_ms / 1000

    def close(self):
        """Disconnect and irreversibly tear down the instrument."""
        if getattr(self, 'visa_handle', None):
            self.visa_handle.close()
        super().close()

    def check_error(self, ret_code):
        """
        Default error checking, raises an error if return code !=0.

        Does not differentiate between warnings or specific error messages.
        Override this function in your driver if you want to add specific
        error messages.

        Args:
            ret_code (int): A Visa error code. See eg:
                https://github.com/hgrecco/pyvisa/blob/master/pyvisa/errors.py

        Raises:
            visa.VisaIOError: if ``ret_code`` indicates a communication
                problem.
        """
        if ret_code != 0:
            raise visa.VisaIOError(ret_code)

    def write_raw(self, cmd):
        """
        Low-level interface to ``visa_handle.write``.

        Args:
            cmd (str): The command to send to the instrument.
        """
        log.debug("Writing to instrument {}: {}".format(self.name, cmd))

        nr_bytes_written, ret_code = self.visa_handle.write(cmd)
        self.check_error(ret_code)

    def ask_raw(self, cmd):
        """
        Low-level interface to ``visa_handle.ask``.

        Args:
            cmd (str): The command to send to the instrument.

        Returns:
            str: The instrument's response.
        """
        log.debug("Querying instrument {}: {}".format(self.name, cmd))
        response = self.visa_handle.query(cmd)
        log.debug(f"Got instrument response: {response}")
        return response

    def snapshot_base(self, update: bool=False,
                      params_to_skip_update: Sequence[str] = None):
        """
        State of the instrument as a JSON-compatible dict.

        Args:
            update (bool): If True, update the state by querying the
                instrument. If False, just use the latest values in memory.
            params_to_skip_update: List of parameter names that will be skipped
                in update even if update is True. This is useful if you have
                parameters that are slow to update but can be updated in a
                different way (as in the qdac)
        Returns:
            dict: base snapshot
        """
        snap = super().snapshot_base(update=update,
                                     params_to_skip_update=params_to_skip_update)

        snap['address'] = self._address
        snap['terminator'] = self._terminator
        snap['timeout'] = self.timeout.get()

        return snap
