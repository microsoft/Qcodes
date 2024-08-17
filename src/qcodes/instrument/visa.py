"""Visa instrument driver based on pyvisa."""

from __future__ import annotations

import logging
import warnings
from importlib.resources import as_file, files
from typing import TYPE_CHECKING, Any, Literal, TypedDict
from weakref import finalize

import pyvisa
import pyvisa.constants as vi_const
import pyvisa.resources
from pyvisa.errors import InvalidSession

import qcodes.validators as vals
from qcodes.logger import get_instrument_logger
from qcodes.utils import DelayedKeyboardInterrupt

from .instrument import Instrument
from .instrument_base import InstrumentBase, InstrumentBaseKWArgs

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from typing_extensions import NotRequired, Unpack

VISA_LOGGER = ".".join((InstrumentBase.__module__, "com", "visa"))

log = logging.getLogger(__name__)


def _close_visa_handle(
    handle: pyvisa.resources.MessageBasedResource, name: str
) -> None:
    try:
        if (
            handle.session is not None
        ):  # pyvisa sets the session of a handle to None when it is closed
            log.info(
                "Closing VISA handle to %s as there are no non weak "
                "references to the instrument.",
                name,
            )

            handle.close()
    except InvalidSession:
        # the resource is already closed
        pass


class VisaInstrumentKWArgs(TypedDict):
    """
    This TypedDict defines the type of the kwargs that can be passed to the VisaInstrument class.
    A subclass of VisaInstrument should take ``**kwargs: Unpack[VisaInstrumentKWArgs]`` as input
    and forward this to the super class to ensure that it can accept all the arguments defined here.

    Consult the documentation of :class:`.VisaInstrument` for more information on the arguments.
    """

    metadata: NotRequired[Mapping[Any, Any] | None]
    """
    Additional static metadata to add to this instrument's JSON snapshot.
    """
    label: NotRequired[str | None]
    """
    Nicely formatted name of the instrument; if None,
    the ``name`` is used.
    """
    terminator: NotRequired[str | None]
    """Read and write termination character(s)."""
    timeout: NotRequired[float]
    "Seconds to allow for responses."
    device_clear: NotRequired[bool]
    "Perform a device clear."
    visalib: NotRequired[str | None]
    """
    Visa backend to use when connecting to this instrument.
    """
    pyvisa_sim_file: NotRequired[str | None]
    """
    Name of a pyvisa-sim yaml file used to simulate the instrument.
    """


class VisaInstrument(Instrument):
    """
    Base class for all instruments using visa connections.

    Args:
        name: What this instrument is called locally.
        address: The visa resource name to use to connect.
        timeout: seconds to allow for responses.  If "unset" will read the value from
           `self.default_timeout`. None means wait forever. Default 5.
        terminator: Read and write termination character(s).
            If unset will use `self.default_terminator`.
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
        **kwargs: Other kwargs are forwarded to the baseclass.

    See help for :class:`.Instrument` for additional information on writing
    instrument subclasses.

    """

    default_terminator: str | None = None
    """
    The default terminator to use if the terminator is not specified when creating the instrument.
    None means use the default terminator from PyVisa.
    """
    default_timeout: float | None = 5
    """
    The default timeout in seconds if the timeout is not specified when creating the instrument.
    None means no timeout e.g. wait forever.
    """

    def __init__(
        self,
        name: str,
        address: str,
        timeout: float | None | Literal["Unset"] = "Unset",
        terminator: str | None | Literal["Unset"] = "Unset",
        device_clear: bool = True,
        visalib: str | None = None,
        pyvisa_sim_file: str | None = None,
        **kwargs: Unpack[InstrumentBaseKWArgs],
    ):
        if terminator == "Unset":
            terminator = self.default_terminator
        if timeout == "Unset":
            timeout = self.default_timeout

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
                visalib = f"{sim_visalib_path!s}@sim"
                (
                    visa_handle,
                    visabackend,
                    resource_manager,
                ) = self._connect_and_handle_error(address, visalib)
        else:
            visa_handle, visabackend, resource_manager = self._connect_and_handle_error(
                address, visalib
            )
        finalize(self, _close_visa_handle, visa_handle, str(self.name))

        self.visabackend: str = visabackend
        self.visa_handle: pyvisa.resources.MessageBasedResource = visa_handle
        """
        The VISA resource used by this instrument.
        """
        self.resource_manager = resource_manager
        """
        The VISA resource manager used by this instrument.
        """
        self.visalib: str | None = visalib
        self._address = address

        if device_clear:
            self.device_clear()

        self.set_terminator(terminator)
        self.timeout.set(timeout)

    def _connect_and_handle_error(
        self, address: str, visalib: str | None
    ) -> tuple[pyvisa.resources.MessageBasedResource, str, pyvisa.ResourceManager]:
        try:
            visa_handle, visabackend, resource_manager = self._open_resource(
                address, visalib
            )
        except Exception as e:
            self.visa_log.exception(f"Could not connect at {address}")
            self.close()
            raise e
        return visa_handle, visabackend, resource_manager

    def _open_resource(
        self, address: str, visalib: str | None
    ) -> tuple[pyvisa.resources.MessageBasedResource, str, pyvisa.ResourceManager]:
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

        return resource, visabackend, resource_manager

    def set_address(self, address: str) -> None:
        """
        Set the address for this instrument.

        Args:
            address: The visa resource name to use to connect. The address
                should be the actual address and just that. If you wish to
                change the backend for VISA, use the self.visalib attribute
                (and then call this function).
        """
        resource, visabackend, resource_manager = self._open_resource(
            address, self.visalib
        )
        self.visa_handle = resource
        self._address = address
        self.visabackend = visabackend
        self.resource_manager = resource_manager

    def device_clear(self) -> None:
        """Clear the buffers of the device"""

        # Serial instruments have a separate flush method to clear
        # their buffers which behaves differently to clear. This is
        # particularly important for instruments which do not support
        # SCPI commands.

        # Simulated instruments do not support a handle clear
        if self.visabackend == "sim":
            return

        flush_operation = (
            vi_const.BufferOperation.discard_read_buffer_no_io
            | vi_const.BufferOperation.discard_write_buffer
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
            self.visa_handle.timeout = float("+inf")
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
        if getattr(self, "visa_handle", None):
            self.visa_handle.close()

        if getattr(self, "visabackend", None) == "sim" and getattr(
            self, "resource_manager", None
        ):
            # The pyvisa-sim visalib has a session attribute but the resource manager is not generic in the
            # visalib type so we cannot get it in a type safe way
            known_sessions = getattr(self.resource_manager.visalib, "sessions", ())

            try:
                this_session = self.resource_manager.session
            except InvalidSession:
                # this may be triggered when the resource has already been closed
                # in that case there is nothing that we can do.
                this_session = None

            session_found = this_session is not None and this_session in known_sessions

            n_sessions = len(known_sessions)
            # if this instrument is the last one or there are no connected instruments its safe to reset the device
            if (session_found and n_sessions == 1) or n_sessions == 0:
                # work around for https://github.com/pyvisa/pyvisa-sim/issues/83
                # see other issues for more context
                # https://github.com/QCoDeS/Qcodes/issues/5356 and
                # https://github.com/pyvisa/pyvisa-sim/issues/82
                try:
                    self.resource_manager.visalib._init()
                except AttributeError:
                    warnings.warn(
                        "The installed version of pyvisa-sim does not have an `_init` method "
                        "in its visa library implementation. Cannot reset simulated instrument state. "
                        "On reconnect the instrument may retain settings set in this session."
                    )

        super().close()

    def write_raw(self, cmd: str) -> None:
        """
        Low-level interface to ``visa_handle.write``.

        Args:
            cmd: The command to send to the instrument.
        """
        with DelayedKeyboardInterrupt(
            context={"instrument": self.name, "reason": "Visa Instrument write"}
        ):
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
        with DelayedKeyboardInterrupt(
            context={"instrument": self.name, "reason": "Visa Instrument ask"}
        ):
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
        snap = super().snapshot_base(
            update=update, params_to_skip_update=params_to_skip_update
        )

        snap["address"] = self._address
        snap["terminator"] = self.visa_handle.read_termination
        snap["read_terminator"] = self.visa_handle.read_termination
        snap["write_terminator"] = self.visa_handle.write_termination
        snap["timeout"] = self.timeout.get()

        return snap
