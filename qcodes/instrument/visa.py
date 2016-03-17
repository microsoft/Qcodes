import visa

from .base import Instrument
from .server import ask_server, write_server


class VisaInstrument(Instrument):
    '''
    Base class for all instruments using visa connections

    name: what this instrument is called locally
    address: the visa resource name. see eg:
        http://pyvisa.readthedocs.org/en/stable/names.html
        (can be changed later with set_address)
    server_name: name of the InstrumentServer to use. By default
        uses 'VisaServer', ie all visa instruments go on the same
        server, but you can provide any other string, or None to
        not use a server (not recommended, then it cannot be used
        with subprocesses like background Loop's)
    timeout: seconds to allow for responses (default 5)
        (can be changed later with set_timeout)
    terminator: the read termination character(s) to expect
        (can be changed later with set_terminator)
    '''
    def __init__(self, name, address=None, server_name='',
                 timeout=5, terminator='', **kwargs):
        # only set the io routines if a subclass doesn't override EITHER
        # the sync or the async version, so we preserve the ability of
        # the base Instrument class to convert between sync and async
        if not self._has_action('write'):
            self._write_fn = self._default_write

        if not self._has_action('ask'):
            self._ask_fn = self._default_ask

        self._address = address
        self._timeout = timeout
        self._terminator = terminator

        if server_name == '':
            server_name = self._default_server_name()
        super().__init__(name, server_name=server_name, **kwargs)

    def _default_server_name(self):
        # TODO - figure out if we want multiple servers by default
        return 'VisaServer'

    @ask_server
    def on_connect(self, address, timeout, terminator):
        self.set_address(self._address)
        self.set_timeout(self._timeout)
        self.set_terminator(self._terminator)

    @write_server
    def set_address(self, address):
        resource_manager = visa.ResourceManager()
        self.visa_handle = resource_manager.open_resource(address)

        self.visa_handle.clear()
        self._address = address

    @write_server
    def set_terminator(self, terminator):
        self.visa_handle.read_termination = terminator
        self._terminator = terminator

    @write_server
    def set_timeout(self, timeout):
        # we specify timeout always in seconds, but visa requires milliseconds
        self.visa_handle.timeout = 1000.0 * timeout
        self._timeout = timeout

    def __del__(self):
        '''
        Close the visa handle upon deleting the object
        '''
        if getattr(self, 'visa_handle', None):
            self.visa_handle.close()
        super().__del__()

    def check_error(self, ret_code):
        '''
        Default error checking, raises an error if return code !=0
        does not differentiate between warnings or specific error messages
        overwrite this function in your driver if you want to add specific
        error messages
        '''
        if ret_code != 0:
            raise visa.VisaIOError(ret_code)

    @write_server
    def _default_write(self, cmd):
        # TODO: lock, async
        nr_bytes_written, ret_code = self.visa_handle.write(cmd)
        self.check_error(ret_code)

    @ask_server
    def _default_ask(self, cmd):
        # TODO: lock, async
        return self.visa_handle.ask(cmd)
