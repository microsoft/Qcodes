import visa
import asyncio

from qcodes.instrument.base import BaseInstrument


class VisaInstrument(BaseInstrument):
    def __init__(self, name, address=None,
                 timeout=5, terminator='', **kwargs):
        super().__init__(name, **kwargs)

        self.set_address(address)
        self.set_timeout(timeout)
        self.set_terminator(terminator)

    def set_address(self, address):
        resource_manager = visa.ResourceManager()
        self.visa_handle = resource_manager.open_resource(address)

        self.visa_handle.clear()
        self._address = address

    def set_terminator(self, terminator):
        self.visa_handle.read_termination = terminator
        self._terminator = terminator

    def set_timeout(self, timeout):
        # we specify timeout always in seconds, but visa requires milliseconds
        self.visa_handle.timeout = 1000.0 * timeout
        self._timeout = timeout

    @asyncio.coroutine
    def write_async(self, cmd):
        # TODO: lock, async
        self.visa_handle.write(cmd)

    @asyncio.coroutine
    def ask_async(self, cmd):
        # TODO: lock, async
        return self.visa_handle.ask(cmd)
