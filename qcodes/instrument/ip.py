import socket

from .base import Instrument
from .server import ask_server, write_server


class IPInstrument(Instrument):
    '''
    Bare socket ethernet instrument implementation

    name: what this instrument is called locally
    address: the IP address or domain name, as a string
    port: the IP port, as an integer
        (address and port can be set later with set_address)
    timeout: seconds to allow for responses (default 5)
        (can be set later with set_timeout)
    terminator: character(s) to terminate each send with (default '\n')
        (can be set later with set_terminator)
    persistent: do we leave the socket open between calls? (default True)
    write_confirmation: does the instrument acknowledge writes with some
        response we can read? (default True)

    See help for qcodes.Instrument for information on writing
    instrument subclasses
    '''
    def __init__(self, name, address=None, port=None, timeout=5,
                 terminator='\n', persistent=True, write_confirmation=True,
                 **kwargs):
        # only set the io routines if a subclass doesn't override EITHER
        # the sync or the async version, so we preserve the ability of
        # the base Instrument class to convert between sync and async
        if not self._has_action('write'):
            self._write_fn = self._default_write

        if not self._has_action('ask'):
            self._ask_fn = self._default_ask

        self._address = address
        self._port = port
        self._timeout = timeout
        self._terminator = terminator
        self._persistent = persistent
        self._confirmation = write_confirmation

        self._socket = None

        super().__init__(name, **kwargs)

    @ask_server
    def on_connect(self):
        if self._persistent:
            self._connect()

    @ask_server
    def set_address(self, address=None, port=None):
        if address is not None:
            self._address = address
        elif not hasattr(self, '_address'):
            raise TypeError('This instrument doesn\'t have an address yet, '
                            'you must provide one.')
        if port is not None:
            self._port = port
        elif not hasattr(self, '_port'):
            raise TypeError('This instrument doesn\'t have a port yet, '
                            'you must provide one.')

        if self._persistent:
            self._connect()

    @ask_server
    def _connect(self):
        if self._socket is not None:
            self._disconnect()

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.connect((self._address, self._port))
        self.set_timeout(self._timeout)

    @ask_server
    def _disconnect(self):
        if getattr(self, '_socket', None) is None:
            return

        self._socket.shutdown(socket.SHUT_RDWR)
        self._socket.close()
        self._socket = None

    @write_server
    def set_timeout(self, timeout=None):
        if timeout is not None:
            self._timeout = timeout

        if self._socket is not None:
            self.socket.settimeout(float(self.timeout))

    @write_server
    def set_terminator(self, terminator):
        self._terminator = terminator

    @write_server
    def _send(self, cmd):
        data = cmd + self._terminator
        self._socket.send(data.encode())

    @ask_server
    def _recv(self):
        return self._socket.recv(512).decode()

    def __del__(self):
        self._disconnect()

    @write_server
    def _default_write(self, cmd):
        with EnsureConnection(self):
            self._send(cmd)
            if self._confirmation:
                self._recv()

    @ask_server
    def _default_ask(self, cmd):
        with EnsureConnection(self):
            self._send(cmd)
            return self._recv()


class EnsureConnection:
    def __init__(self, instrument):
        self._instrument = instrument

    def __enter__(self):
        if not self.instrument._persistent or self.instrument._socket is None:
            self.instrument._connect()

    def __exit__(self):
        if not self.instrument._persistent:
            self.instrument._disconnect()
