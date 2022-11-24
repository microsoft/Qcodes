"""Visa instrument driver based on pyvisa."""
from __future__ import annotations

import sys

if sys.version_info >= (3, 9):
    from importlib.resources import as_file, files
else:
    from importlib_resources import as_file, files

import logging
from collections.abc import Sequence
from typing import Any

import pyvisa
import pyvisa.constants as vi_const
import pyvisa.resources

import qcodes.validators as vals
from qcodes.logger import get_instrument_logger
from qcodes.utils import DelayedKeyboardInterrupt

from .instrument import Instrument
from .instrument_base import InstrumentBase

VISA_LOGGER = '.'.join((InstrumentBase.__module__, 'com', 'visa'))

log = logging.getLogger(__name__)


class VisaInstrument(Instrument):

    """
    Base class for all instruments using visa connections.

    Args:
        name: What this instrument is called locally.
        address: The visa resource name to use to connect.
        timeout: seconds to allow for responses. Default 5.
        terminator: Read and write termination character(s).
            If None the terminator will not be set and we
            rely on the defaults from PyVisa. Default None.
        device_clear: Perform a device clear. Default True.
        visalib: Visa backend to use when connecting to this instrument.
            This should be in the form of a string '<pathtofile>@<backend>'.
            Both parts can be omitted and pyvisa will try to infer the
            path to the visa backend file.
            By default the IVI backend is used if found, but '@py' will use the
            ``pyvisa-py`` backend. Note that QCoDeS does not install (or even require)
            ANY backends, it is up to the user to do that. see eg:
            http://pyvisa.readthedocs.org/en/stable/names.html
        metadata: additional static metadata to add to this
            instrument's JSON snapshot.
        pyvisa_sim_file: Name of a pyvisa-sim yaml file used to simulate the instrument.
            The file is expected to be loaded from a python module.
            The file can be given either as only the file name in which case it is loaded
            from ``qcodes.instruments.sims`` or in the format ``module:filename`` e.g.
            ``qcodes.instruments.sims:AimTTi_PL601P.yaml`` in which case it is loaded
            from the supplied module. Note that it is an error to pass both
            ``pyvisa_sim_file`` and ``visalib``.

    See help for :class:`.Instrument` for additional information on writing
    instrument subclasses.

    """

    def __init__(
        self,
        name: str,
        address: str,
        timeout: float = 5,
        terminator: str | None = None,
        device_clear: bool = True,
        visalib: str | None = None,
        pyvisa_sim_file: str | None = None,
        **kwargs: Any,
    ):

        super().__init__(name, **kwargs)
        self.visa_log = get_instrument_logger(self, VISA_LOGGER)

        self.add_parameter(
            "timeout",
            get_cmd=self._get_visa_timeout,
            set_cmd=self._set_visa_timeout,
            unit="s",
            vals=vals.MultiType(vals.Numbers(min_value=0), vals.Enum(None)),
        )

        if visalib is not None and pyvisa_sim_file is not None:
            raise RuntimeError(
                "It's an error to supply both visalib and pyvisa_sim_file as "
                "arguments to a VISA instrument"
            )
        if pyvisa_sim_file is not None:
            if ":" in pyvisa_sim_file:
                module, pyvisa_sim_file = pyvisa_sim_file.split(":")
            else:
                module = "qcodes.instrument.sims"
            traversable_handle = files(module) / pyvisa_sim_file
            with as_file(traversable_handle) as sim_visalib_path:
                if not sim_visalib_path.exists():
                    raise FileNotFoundError(
                        "Pyvisa-sim yaml file "
                        "could not be found. Trying to load "
                        f"file {pyvisa_sim_file} from module: {module}"
                    )
                visalib = f"{str(sim_visalib_path)}@sim"
                visa_handle, visabackend = self._connect_and_handle_error(
                    address, visalib
                )
        else:
            visa_handle, visabackend = self._connect_and_handle_error(address, visalib)

        self.visabackend: str = visabackend
        self.visa_handle: pyvisa.resources.MessageBasedResource = visa_handle
        """
        The VISA resource used by this instrument.
        """
        self.visalib: str | None = visalib
        self._address = address

        if device_clear:
            self.device_clear()

        self.set_terminator(terminator)
        self.timeout.set(timeout)

    def _connect_and_handle_error(
        self, address: str, visalib: str | None
    ) -> tuple[pyvisa.resources.MessageBasedResource, str]:
        try:
            visa_handle, visabackend = self._open_resource(address, visalib)
        except Exception as e:
            self.visa_log.exception(f"Could not connect at {address}")
            self.close()
            raise e
        return visa_handle, visabackend

    def _open_resource(
        self, address: str, visalib: str | None
    ) -> tuple[pyvisa.resources.MessageBasedResource, str]:

        # in case we're changing the address - close the old handle first
        if getattr(self, "visa_handle", None):
            self.visa_handle.close()

        if visalib is not None:
            self.visa_log.info(
                f"Opening PyVISA Resource Manager with visalib: {visalib}"
            )
            resource_manager = pyvisa.ResourceManager(visalib)
            visabackend = visalib.split("@")[1]
        else:
            self.visa_log.info("Opening PyVISA Resource Manager with default backend.")
            resource_manager = pyvisa.ResourceManager()
            visabackend = "ivi"

        self.visa_log.info(f"Opening PyVISA resource at address: {address}")
        resource = resource_manager.open_resource(address)
        if not isinstance(resource, pyvisa.resources.MessageBasedResource):
            resource.close()
            raise TypeError("QCoDeS only support MessageBasedResource Visa resources")

        return resource, visabackend

    def set_address(self, address: str) -> None:
        """
        Set the address for this instrument.

        Args:
            address: The visa resource name to use to connect. The address
                should be the actual address and just that. If you wish to
                change the backend for VISA, use the self.visalib attribute
                (and then call this function).
        """
        resource, visabackend = self._open_resource(address, self.visalib)
        self.visa_handle = resource
        self._address = address
        self.visabackend = visabackend

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

    def set_terminator(self, terminator: str | None) -> None:
        r"""
        Change the read terminator to use.

        Args:
            terminator: Character(s) to look for at the end of a read and
                to end each write command with.
                eg. ``\r\n``. If None the terminator will not be set.
        """
        if terminator is not None:
            self.visa_handle.write_termination = terminator
            self.visa_handle.read_termination = terminator

    def _set_visa_timeout(self, timeout: float | None) -> None:
        # according to https://pyvisa.readthedocs.io/en/latest/introduction/resources.html#timeout
        # both float('+inf') and None are accepted as meaning infinite timeout
        # however None does not pass the typechecking in 1.11.1
        if timeout is None:
            self.visa_handle.timeout = float('+inf')
        else:
            # pyvisa uses milliseconds but we use seconds
            self.visa_handle.timeout = timeout * 1000.0

    def _get_visa_timeout(self) -> float | None:

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

    def snapshot_base(
        self,
        update: bool | None = True,
        params_to_skip_update: Sequence[str] | None = None,
    ) -> dict[Any, Any]:
        """
        State of the instrument as a JSON-compatible dict (everything that
        the custom JSON encoder class :class:`.NumpyJSONEncoder`
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

        snap["address"] = self._address
        snap["terminator"] = self.visa_handle.read_termination
        snap["read_terminator"] = self.visa_handle.read_termination
        snap["write_terminator"] = self.visa_handle.write_termination
        snap["timeout"] = self.timeout.get()

        return snap
