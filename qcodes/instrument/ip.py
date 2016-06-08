import socket

from .base import Instrument


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
    '''
    def __init__(self, name, address=None, port=None, timeout=5,
                 terminator='\n', persistent=True, write_confirmation=True,
                 **kwargs):
        super().__init__(name, **kwargs)

        self._address = address
        self._port = port
        self._timeout = timeout
        self._terminator = terminator
        self._confirmation = write_confirmation

        self._ensure_connection = EnsureConnection(self)

        self._socket = None

        self.set_persistent(persistent)

    @classmethod
    def default_server_name(cls, **kwargs):
        return 'IPInstruments'

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

        self._disconnect()
        self.set_persistent(self._persistent)

    def set_persistent(self, persistent):
        self._persistent = persistent
        if persistent:
            self._connect()
        else:
            self._disconnect()

    def _connect(self):
        if self._socket is not None:
            self._disconnect()

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.connect((self._address, self._port))
        self.set_timeout(self._timeout)

    def _disconnect(self):
        if getattr(self, '_socket', None) is None:
            return

        self._socket.shutdown(socket.SHUT_RDWR)
        self._socket.close()
        self._socket = None

    def set_timeout(self, timeout=None):
        if timeout is not None:
            self._timeout = timeout

        if self._socket is not None:
            self.socket.settimeout(float(self.timeout))

    def set_terminator(self, terminator):
        self._terminator = terminator

    def _send(self, cmd):
        data = cmd + self._terminator
        self._socket.send(data.encode())

    def _recv(self):
        return self._socket.recv(512).decode()

    def close(self):
        self._disconnect()
        super().close()

    def write_raw(self, cmd):
        """Low-level interface to send a command that gets no response."""
        with self._ensure_connection:
            self._send(cmd)
            if self._confirmation:
                self._recv()

    def ask_raw(self, cmd):
        """Low-level interface to send a command an read a response."""
        with self._ensure_connection:
            self._send(cmd)
            return self._recv()

    def snapshot_base(self, update=False):
        snap = super().snapshot_base(update=update)

        snap['port'] = self._port
        snap['confirmation'] = self._confirmation
        snap['address'] = self._address
        snap['terminator'] = self._terminator
        snap['timeout'] = self._timeout
        snap['persistent'] = self._persistent

        return snap


class EnsureConnection:
    def __init__(self, instrument):
        self._instrument = instrument

    def __enter__(self):
        if not self.instrument._persistent or self.instrument._socket is None:
            self.instrument._connect()

    def __exit__(self):
        if not self.instrument._persistent:
            self.instrument._disconnect()
