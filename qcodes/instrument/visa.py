"""Visa instrument driver based on pyvisa."""
from typing import Sequence, Optional, Dict, Union, Any, cast
import warnings
import logging
from packaging.version import Version

import pyvisa as visa
import pyvisa.constants as vi_const
import pyvisa.resources

from .base import Instrument, InstrumentBase

import qcodes.utils.validators as vals
from qcodes.utils.deprecate import deprecate
from qcodes.logger.instrument_logger import get_instrument_logger
from qcodes.utils.delaykeyboardinterrupt import DelayedKeyboardInterrupt

VISA_LOGGER = '.'.join((InstrumentBase.__module__, 'com', 'visa'))

log = logging.getLogger(__name__)


class VisaInstrument(Instrument):

    """
    Base class for all instruments using visa connections.

    Args:
        name: What this instrument is called locally.
        address: The visa resource name to use to connect.
        timeout: seconds to allow for responses. Default 5.
        terminator: Read termination character(s) to look for. Default ``''``.
        device_clear: Perform a device clear. Default True.
        visalib: Visa backend to use when connecting to this instrument.
            This should be in the form of a string '@<backend>'.
            By default the NI backend is used, but '@py' will use the
            ``pyvisa-py`` backend. Note that QCoDeS does not install (or even require)
            ANY backends, it is up to the user to do that. see eg:
            http://pyvisa.readthedocs.org/en/stable/names.html
        metadata: additional static metadata to add to this
            instrument's JSON snapshot.

    See help for :class:`.Instrument` for additional information on writing
    instrument subclasses.

    Attributes:
        visa_handle (pyvisa.resources.Resource): The communication channel.
    """

    def __init__(self, name: str, address: str, timeout: Union[int, float] = 5,
                 terminator: str = '', device_clear: bool = True,
                 visalib: Optional[str] = None, **kwargs: Any):

        super().__init__(name, **kwargs)
        self.visa_log = get_instrument_logger(self, VISA_LOGGER)
        self.visabackend: str
        self.visa_handle: visa.resources.MessageBasedResource
        self.visalib: Optional[str]

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

        try:
            self.set_address(address)
        except Exception as e:
            self.visa_log.info(f"Could not connect at {address}")
            self.close()
            raise e

        if device_clear:
            self.device_clear()

        self.set_terminator(terminator)
        self.timeout.set(timeout)

    def set_address(self, address: str) -> None:
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
            self.visa_log.info('Opening PyVISA Resource Manager with visalib:'
                          ' {}'.format(self.visalib))
            resource_manager = visa.ResourceManager(self.visalib)
            self.visabackend = self.visalib.split('@')[1]
        else:
            self.visa_log.info('Opening PyVISA Resource Manager with default'
                          ' backend.')
            resource_manager = visa.ResourceManager()
            self.visabackend = 'ni'

        self.visa_log.info(f'Opening PyVISA resource at address: {address}')
        resource = resource_manager.open_resource(address)
        if not isinstance(resource, visa.resources.MessageBasedResource):
            raise TypeError("QCoDeS only support MessageBasedResource "
                            "Visa resources")
        self.visa_handle = resource
        self._address = address

    def device_clear(self) -> None:
        """Clear the buffers of the device"""

        # Serial instruments have a separate flush method to clear
        # their buffers which behaves differently to clear. This is
        # particularly important for instruments which do not support
        # SCPI commands.

        # Simulated instruments do not support a handle clear
        if self.visabackend == 'sim':
            return

        flush_operation = (
                vi_const.BufferOperation.discard_read_buffer_no_io |
                vi_const.BufferOperation.discard_write_buffer
        )

        if isinstance(self.visa_handle, pyvisa.resources.SerialInstrument):
            self.visa_handle.flush(flush_operation)
        else:
            self.visa_handle.clear()

    def set_terminator(self, terminator: str) -> None:
        r"""
        Change the read terminator to use.

        Args:
            terminator: Character(s) to look for at the end of a read.
                eg. ``\r\n``.
        """
        self.visa_handle.write_termination = terminator
        self.visa_handle.read_termination = terminator
        self._terminator = terminator

        if self.visabackend == 'sim':
            self.visa_handle.write_termination = terminator

    def _set_visa_timeout(self, timeout: Optional[float]) -> None:
        # according to https://pyvisa.readthedocs.io/en/latest/introduction/resources.html#timeout
        # both float('+inf') and None are accepted as meaning infinite timeout
        # however None does not pass the typechecking in 1.11.1
        if timeout is None:
            self.visa_handle.timeout = float('+inf')
        else:
            # pyvisa uses milliseconds but we use seconds
            self.visa_handle.timeout = timeout * 1000.0

    def _get_visa_timeout(self) -> Optional[float]:

        timeout_ms = self.visa_handle.timeout
        if timeout_ms is None:
            return None
        else:
            # pyvisa uses milliseconds but we use seconds
            return timeout_ms / 1000

    def close(self) -> None:
        """Disconnect and irreversibly tear down the instrument."""
        if getattr(self, 'visa_handle', None):
            self.visa_handle.close()
        super().close()

    @deprecate(reason="pyvisa already checks the error code itself")
    def check_error(self, ret_code: int) -> None:
        """
        Default error checking, raises an error if return code ``!=0``.
        Does not differentiate between warnings or specific error messages.
        Override this function in your driver if you want to add specific
        error messages.

        Args:
            ret_code: A Visa error code. See eg:
                https://github.com/hgrecco/pyvisa/blob/master/pyvisa/errors.py

        Raises:
            visa.VisaIOError: if ``ret_code`` indicates a communication
                problem.
        """
        if ret_code != 0:
            raise visa.VisaIOError(ret_code)

    def write_raw(self, cmd: str) -> None:
        """
        Low-level interface to ``visa_handle.write``.

        Args:
            cmd: The command to send to the instrument.
        """
        with DelayedKeyboardInterrupt():
            self.visa_log.debug(f"Writing: {cmd}")
            self.visa_handle.write(cmd)

    def ask_raw(self, cmd: str) -> str:
        """
        Low-level interface to ``visa_handle.ask``.

        Args:
            cmd: The command to send to the instrument.

        Returns:
            str: The instrument's response.
        """
        with DelayedKeyboardInterrupt():
            self.visa_log.debug(f"Querying: {cmd}")
            response = self.visa_handle.query(cmd)
            self.visa_log.debug(f"Response: {response}")
        return response

    def snapshot_base(self, update: Optional[bool] = True,
                      params_to_skip_update: Optional[Sequence[str]] = None
                      ) -> Dict[Any, Any]:
        """
        State of the instrument as a JSON-compatible dict (everything that
        the custom JSON encoder class :class:`qcodes.utils.helpers.NumpyJSONEncoder`
        supports).

        Args:
            update: If True, update the state by querying the
                instrument. If None only update if the state is known to be
                invalid. If False, just use the latest values in memory and
                never update.
            params_to_skip_update: List of parameter names that will be skipped
                in update even if update is True. This is useful if you have
                parameters that are slow to update but can be updated in a
                different way (as in the qdac). If you want to skip the
                update of certain parameters in all snapshots, use the
                ``snapshot_get``  attribute of those parameters instead.
        Returns:
            dict: base snapshot
        """
        snap = super().snapshot_base(update=update,
                                     params_to_skip_update=params_to_skip_update)

        snap['address'] = self._address
        snap['terminator'] = self._terminator
        snap['timeout'] = self.timeout.get()

        return snap
