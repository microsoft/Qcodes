import socket
import asyncio

from .base import Instrument


class IPInstrument(Instrument):
    def __init__(self, name, address=None, port=None,
                 timeout=5, terminator='\n', **kwargs):
        super().__init__(name, **kwargs)

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((address, port))

        self.set_timeout(timeout)
        self.terminator = terminator

    def set_timeout(self, timeout):
        self.socket.settimeout(float(timeout))
        self._timeout = timeout

    @asyncio.coroutine
    def write_async(self, cmd):
        data = cmd + self.terminator
        self.socket.send(data.encode())

    @asyncio.coroutine
    def ask_async(self, cmd):
        data = cmd + self.terminator
        self.socket.send(data.encode())
        # TODO: async? what's the 512?
        # TODO: and why did athena *always* have a recv, even if there's
        # no return?
        # if this is an instrument-specific thing (like some instruments always
        # reply "OK" to set commands) then these can override write_async
        # to ask_async with response validation
        # TODO: does this need a lock too? probably...
        return self.socket.recv(512).decode()
