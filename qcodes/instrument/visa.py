"""Visa instrument driver based on pyvisa."""
import visa

from .base import Instrument
import qcodes.utils.validators as vals


class VisaInstrument(Instrument):

    """
    Base class for all instruments using visa connections.

    Args:
        name (str): What this instrument is called locally.

        address (str): The visa resource name to use to connect.
            Optionally includes '@<backend>' at the end. For example,
            'ASRL2' will open COM2 with the default NI backend, but
            'ASRL2@py' will open COM2 using pyvisa-py. Note that qcodes
            does not install (or even require) ANY backends, it is up to
            the user to do that. see eg:
            http://pyvisa.readthedocs.org/en/stable/names.html

        timeout (number): seconds to allow for responses. Default 5.

        terminator: Read termination character(s) to look for. Default ''.

        server_name (str): Name of the InstrumentServer to use. By default
            uses 'GPIBServer' for all GPIB instruments, 'SerialServer' for
            serial port instruments, and 'VisaServer' for all others.

            Use ``None`` to run locally - but then this instrument will not
            work with qcodes Loops or other multiprocess procedures.

        metadata (Optional[Dict]): additional static metadata to add to this
            instrument's JSON snapshot.

    See help for ``qcodes.Instrument`` for additional information on writing
    instrument subclasses.

    Attributes:
        visa_handle (pyvisa.resources.Resource): The communication channel.
    """

    def __init__(self, name, address=None, timeout=5, terminator='', **kwargs):
        super().__init__(name, **kwargs)

        self.add_parameter('timeout',
                           get_cmd=self._get_visa_timeout,
                           set_cmd=self._set_visa_timeout,
                           unit='s',
                           vals=vals.MultiType(vals.Numbers(min_value=0),
                                               vals.Enum(None)))

        self.set_address(address)
        self.set_terminator(terminator)
        self.timeout.set(timeout)

    def set_address(self, address):
        """
        Change the address for this instrument.

        Args:
            address: The visa resource name to use to connect.
                Optionally includes '@<backend>' at the end. For example,
                'ASRL2' will open COM2 with the default NI backend, but
                'ASRL2@py' will open COM2 using pyvisa-py. Note that qcodes
                does not install (or even require) ANY backends, it is up to
                the user to do that.
                see eg: http://pyvisa.readthedocs.org/en/stable/names.html
        """
        # in case we're changing the address - close the old handle first
        if getattr(self, 'visa_handle', None):
            self.visa_handle.close()

        if address and '@' in address:
            address, visa_library = address.split('@')
            resource_manager = visa.ResourceManager('@' + visa_library)
        else:
            resource_manager = visa.ResourceManager()

        self.visa_handle = resource_manager.open_resource(address)

        self.visa_handle.clear()
        self._address = address

    def set_terminator(self, terminator):
        r"""
        Change the read terminator to use.

        Args:
            terminator (str): Character(s) to look for at the end of a read.
                eg. '\r\n'.
        """
        self.visa_handle.read_termination = terminator
        self._terminator = terminator

    def _set_visa_timeout(self, timeout):
        if timeout is None:
            self.visa_handle.timeout = None
        else:
            # pyvisa uses milliseconds but we use seconds
            self.visa_handle.timeout = timeout * 1000.0

    def _get_visa_timeout(self):
        timeout_ms = self.visa_handle.timeout
        if timeout_ms is None:
            return None
        else:
            # pyvisa uses milliseconds but we use seconds
            return timeout_ms / 1000

    def close(self):
        """Disconnect and irreversibly tear down the instrument."""
        if getattr(self, 'visa_handle', None):
            self.visa_handle.close()
        super().close()

    def check_error(self, ret_code):
        """
        Default error checking, raises an error if return code !=0.

        Does not differentiate between warnings or specific error messages.
        Override this function in your driver if you want to add specific
        error messages.

        Args:
            ret_code (int): A Visa error code. See eg:
                https://github.com/hgrecco/pyvisa/blob/master/pyvisa/errors.py

        Raises:
            visa.VisaIOError: if ``ret_code`` indicates a communication
                problem.
        """
        if ret_code != 0:
            raise visa.VisaIOError(ret_code)

    def write_raw(self, cmd):
        """
        Low-level interface to ``visa_handle.write``.

        Args:
            cmd (str): The command to send to the instrument.
        """
        nr_bytes_written, ret_code = self.visa_handle.write(cmd)
        self.check_error(ret_code)

    def ask_raw(self, cmd):
        """
        Low-level interface to ``visa_handle.ask``.

        Args:
            cmd (str): The command to send to the instrument.

        Returns:
            str: The instrument's response.
        """
        return self.visa_handle.ask(cmd)

    def snapshot_base(self, update=False):
        """
        State of the instrument as a JSON-compatible dict.

        Args:
            update (bool): If True, update the state by querying the
                instrument. If False, just use the latest values in memory.

        Returns:
            dict: base snapshot
        """
        snap = super().snapshot_base(update=update)

        snap['address'] = self._address
        snap['terminator'] = self._terminator
        snap['timeout'] = self.timeout.get()

        return snap
