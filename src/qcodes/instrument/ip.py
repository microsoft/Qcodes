"""Ethernet instrument driver class based on sockets."""

from __future__ import annotations

import logging
import socket
from typing import TYPE_CHECKING, Any

from .instrument import Instrument

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType

    from typing_extensions import Unpack

    from .instrument_base import InstrumentBaseKWArgs

log = logging.getLogger(__name__)


class IPInstrument(Instrument):
    r"""
    Bare socket ethernet instrument implementation. Use of `VisaInstrument`
    is promoted instead of this.

    Args:
        name: What this instrument is called locally.
        address: The IP address or name. If not given on
            construction, must be provided before any communication.
        port: The IP port. If not given on construction, must
            be provided before any communication.
        timeout: Seconds to allow for responses. Default 5.
        terminator: Character(s) to terminate each send. Default '\n'.
        persistent: Whether to leave the socket open between calls.
            Default True.
        write_confirmation: Whether the instrument acknowledges writes
            with some response we should read. Default True.
        **kwargs: Forwarded to the base class.

    See help for ``qcodes.Instrument`` for additional information on writing
    instrument subclasses.
    """

    def __init__(
        self,
        name: str,
        address: str | None = None,
        port: int | None = None,
        timeout: float = 5,
        terminator: str = "\n",
        persistent: bool = True,
        write_confirmation: bool = True,
        **kwargs: Unpack[InstrumentBaseKWArgs],
    ):
        super().__init__(name, **kwargs)

        self._address = address
        self._port = port
        self._timeout = timeout
        self._terminator = terminator
        self._confirmation = write_confirmation

        self._ensure_connection = EnsureConnection(self)
        self._buffer_size = 1400

        self._socket: socket.socket | None = None

        self.set_persistent(persistent)

    def set_address(self, address: str | None = None, port: int | None = None) -> None:
        """
        Change the IP address and/or port of this instrument.

        Args:
            address: The IP address or name.
            port: The IP port.
        """
        if address is not None:
            self._address = address
        elif not hasattr(self, "_address"):
            raise TypeError(
                "This instrument doesn't have an address yet, you must provide one."
            )
        if port is not None:
            self._port = port
        elif not hasattr(self, "_port"):
            raise TypeError(
                "This instrument doesn't have a port yet, you must provide one."
            )

        self._disconnect()
        self.set_persistent(self._persistent)

    def set_persistent(self, persistent: bool) -> None:
        """
        Change whether this instrument keeps its socket open between calls.

        Args:
            persistent: Set True to keep the socket open all the time.
        """
        self._persistent = persistent
        if persistent:
            self._connect()
        else:
            self._disconnect()

    def flush_connection(self) -> None:
        self._recv()

    def _connect(self) -> None:
        if self._socket is not None:
            self._disconnect()

        try:
            log.info("Opening socket")
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            log.info(f"Connecting socket to {self._address}:{self._port}")
            self._socket.connect((self._address, self._port))
            self.set_timeout(self._timeout)
        except ConnectionRefusedError:
            log.warning("Socket connection failed")
            if self._socket is not None:
                self._socket.close()
            self._socket = None
            raise

    def _disconnect(self) -> None:
        if self._socket is None:
            return
        log.info("Socket shutdown")
        self._socket.shutdown(socket.SHUT_RDWR)
        log.info("Socket closing")
        self._socket.close()
        log.info("Socket closed")
        self._socket = None

    def set_timeout(self, timeout: float) -> None:
        """
        Change the read timeout for the socket.

        Args:
            timeout: Seconds to allow for responses.
        """
        self._timeout = timeout

        if self._socket is not None:
            self._socket.settimeout(float(self._timeout))

    def set_terminator(self, terminator: str) -> None:
        r"""
        Change the write terminator to use.

        Args:
            terminator: Character(s) to terminate each send.
                Default '\n'.
        """
        self._terminator = terminator

    def _send(self, cmd: str) -> None:
        if self._socket is None:
            raise RuntimeError(f"IPInstrument {self.name} is not connected")
        data = cmd + self._terminator
        log.debug(f"Writing {data} to instrument {self.name}")
        self._socket.sendall(data.encode())

    def _recv(self) -> str:
        if self._socket is None:
            raise RuntimeError(f"IPInstrument {self.name} is not connected")
        result = self._socket.recv(self._buffer_size)
        log.debug(f"Got {result!r} from instrument {self.name}")
        if result == b"":
            log.warning("Got empty response from Socket recv() Connection broken.")
        return result.decode()

    def close(self) -> None:
        """Disconnect and irreversibly tear down the instrument."""
        self._disconnect()
        super().close()

    def write_raw(self, cmd: str) -> None:
        """
        Low-level interface to send a command that gets no response.

        Args:
            cmd: The command to send to the instrument.
        """

        with self._ensure_connection:
            self._send(cmd)
            if self._confirmation:
                self._recv()

    def ask_raw(self, cmd: str) -> str:
        """
        Low-level interface to send a command an read a response.

        Args:
            cmd: The command to send to the instrument.

        Returns:
            The instrument's string response.
        """
        with self._ensure_connection:
            self._send(cmd)
            return self._recv()

    def snapshot_base(
        self,
        update: bool | None = False,
        params_to_skip_update: Sequence[str] | None = None,
    ) -> dict[Any, Any]:
        """
        State of the instrument as a JSON-compatible dict (everything that
        the custom JSON encoder class
        :class:`.NumpyJSONEncoder`
        supports).

        Args:
            update: If True, update the state by querying the
                instrument. If None only update if the state is known to be
                invalid. If False, just use the latest values in memory and
                never update.
            params_to_skip_update: List of parameter names that will be
                skipped in update even if update is True. This is useful
                if you have parameters that are slow to update but can
                be updated in a different way (as in the qdac). If you
                want to skip the update of certain parameters in all
                snapshots, use the `snapshot_get`  attribute of those
                parameters: instead.

        Returns:
            dict: base snapshot
        """
        snap = super().snapshot_base(
            update=update, params_to_skip_update=params_to_skip_update
        )

        snap["port"] = self._port
        snap["confirmation"] = self._confirmation
        snap["address"] = self._address
        snap["terminator"] = self._terminator
        snap["timeout"] = self._timeout
        snap["persistent"] = self._persistent

        return snap


class EnsureConnection:
    """
    Context manager to ensure an instrument is connected when needed.

    Uses ``instrument._persistent`` to determine whether or not to close
    the connection immediately on completion.

    Args:
        instrument: the instance to connect.
    """

    def __init__(self, instrument: IPInstrument):
        self.instrument = instrument

    def __enter__(self) -> None:
        """Make sure we connect when entering the context."""
        if not self.instrument._persistent or self.instrument._socket is None:
            self.instrument._connect()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Possibly disconnect on exiting the context."""
        if not self.instrument._persistent:
            self.instrument._disconnect()
