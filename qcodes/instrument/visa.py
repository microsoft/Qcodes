import visa
import asyncio

from .base import Instrument


class VisaInstrument(Instrument):
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

    def check_error(self, ret_code):
        '''
        Default error checking, raises an error if return code !=0
        does not differentiate between warnings or specific error messages
        overwrite this function in your driver if you want to add specific
        error messages
        '''
        if ret_code != 0:
            raise visa.VisaIOError(ret_code[1])

    @asyncio.coroutine
    def write_async(self, cmd):
        # TODO: lock, async
        # TODO: return value does not yet get passed back to the notebook but
        # get's caught somewhere in the parameter functions
        nr_bytes_written, ret_code = self.visa_handle.write(cmd)
        self.check_error(ret_code)
        return ret_code

    @asyncio.coroutine
    def ask_async(self, cmd):
        # TODO: lock, async
        return self.visa_handle.ask(cmd)
