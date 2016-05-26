import visa
import logging

from .base import Instrument


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

    See help for qcodes.Instrument for information on writing
    instrument subclasses
    '''
    def __init__(self, name, address=None, timeout=5, terminator='', **kwargs):
        super().__init__(name, **kwargs)

        self.set_address(address)
        self.set_timeout(timeout)
        self.set_terminator(terminator)

    @classmethod
    def default_server_name(cls, **kwargs):
        upper_address = kwargs.get('address', '').upper()
        if 'GPIB' in upper_address:
            return 'GPIBServer'
        elif 'ASRL' in upper_address:
            return 'SerialServer'

        # TODO - any others to break out by default?
        # break out separate GPIB or serial connections?
        return 'VisaServer'

    def get_idn(self):
        """Parse a standard VISA '*IDN?' response into an ID dict."""
        idstr = self.ask('*IDN?')
        try:
            # form is supposed to be comma-separated, but we've seen
            # other separators occasionally
            for separator in ',;:':
                # split into no more than 4 parts, so we don't lose info
                idparts = [p.strip() for p in idstr.split(separator, 3)]
                if len(idparts) > 1:
                    break
            # in case parts at the end are missing, fill in None
            if len(idparts) < 4:
                idparts += [None] * (4 - len(idparts))
        except:
            logging.warn('Error interpreting *IDN? response ' + repr(idstr))
            idparts = [None, None, None, None]

        # some strings include the word 'model' at the front of model
        if str(idparts[1]).lower().startswith('model'):
            idparts[1] = str(idparts[1])[5:].strip()

        return dict(zip(('vendor', 'model', 'serial', 'firmware'), idparts))

    def set_address(self, address):
        '''
        change the address (visa resource name) for this instrument
        see eg: http://pyvisa.readthedocs.org/en/stable/names.html
        '''
        # in case we're changing the address - close the old handle first
        # but not by calling self.close() because that tears down the whole
        # instrument!
        if getattr(self, 'visa_handle', None):
            self.visa_handle.close()

        resource_manager = visa.ResourceManager()
        self.visa_handle = resource_manager.open_resource(address)

        self.visa_handle.clear()
        self._address = address

    def set_terminator(self, terminator):
        '''
        change the read terminator (string, eg '\r\n')
        '''
        self.visa_handle.read_termination = terminator
        self._terminator = terminator

    def set_timeout(self, timeout):
        '''
        change the read timeout (seconds)
        '''
        # visa requires milliseconds
        self.visa_handle.timeout = 1000.0 * timeout
        self._timeout = timeout

    def close(self):
        if getattr(self, 'visa_handle', None):
            self.visa_handle.close()
        super().close()

    def check_error(self, ret_code):
        '''
        Default error checking, raises an error if return code !=0
        does not differentiate between warnings or specific error messages
        overwrite this function in your driver if you want to add specific
        error messages
        '''
        if ret_code != 0:
            raise visa.VisaIOError(ret_code)

    def write(self, cmd):
        try:
            nr_bytes_written, ret_code = self.visa_handle.write(cmd)
            self.check_error(ret_code)
        except Exception as e:
            e.args = e.args + ('writing ' + repr(cmd) + ' to ' + repr(self),)
            raise e

    def ask(self, cmd):
        try:
            return self.visa_handle.ask(cmd)
        except Exception as e:
            e.args = e.args + ('asking ' + repr(cmd) + ' to ' + repr(self),)
            raise e

    def snapshot_base(self, update=False):
        snap = super().snapshot_base(update=update)

        snap['address'] = self._address
        snap['terminator'] = self._terminator
        snap['timeout'] = self._timeout

        return snap
