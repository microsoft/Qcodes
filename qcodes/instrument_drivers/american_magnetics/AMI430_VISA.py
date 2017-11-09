# if this actually works...

import time
import warnings

import qcodes.utils.validators as vals
from qcodes.instrument.ip import IPInstrument
from qcodes.instrument.visa import VisaInstrument
from qcodes.instrument_drivers.american_magnetics.AMI430 import AMI430
from qcodes.utils.helpers import DelegateAttributes, strip_attrs, full_class


class IPToVisa(VisaInstrument, IPInstrument):
    """
    Class to inject an VisaInstrument like behaviour in an
    IPInstrument that we'd like to use as a VISAInstrument with the
    simulation backend.
    The idea is to inject this class just before the IPInstrument in
    the MRO. To avoid IPInstrument to ever take any effect, we avoid
    calling super() up to the IPInstrument class and therefore explicitly
    re-implement the whole inheritance chain in this every method that
    cals super. These methods include __init__, close

    ONLY FOR TESTING/SIMULATION PURPOSES!
    DO NOT USE THIS FOR ACTUAL INSTRUMENT CONTROL
    """

    def __init__(self, name, address, port, visalib,
                 metadata=None, device_clear=False, terminator='\n',
                 timeout=3, testing=False, **kwargs):

        # I'd like to get rid of this attribute ASAP
        self._testing = testing

        ##################################################
        # __init__ of Instrument part 1

        self._t0 = time.time()
        if kwargs.pop('server_name', False):
            warnings.warn("server_name argument not supported any more",
                          stacklevel=0)

        ##################################################
        # __init__ of BaseInstrument
        self.name = str(name)
        self.parameters = {}
        self.functions = {}
        self.submodules = {}

        ##################################################
        # __init__ of Metadatable
        self.metadata = {}
        self.load_metadata(metadata or {})

        ##################################################
        # __init__ of Instrument part 2

        self.add_parameter('IDN', get_cmd=self.get_idn,
                           vals=vals.Anything())

        self._meta_attrs = ['name']

        self.record_instance(self)

        ##################################################
        # __init__ of VisaInstrument

        self.add_parameter('timeout',
                           get_cmd=self._get_visa_timeout,
                           set_cmd=self._set_visa_timeout,
                           unit='s',
                           vals=vals.MultiType(vals.Numbers(min_value=0),
                                               vals.Enum(None)))

        # auxiliary VISA library to use for mocking
        self.visalib = visalib
        self.visabackend = None

        self.set_address(address)
        if device_clear:
            self.device_clear()

        self.set_terminator(terminator)
        self.timeout.set(timeout)

    def close(self):
        """Disconnect and irreversibly tear down the instrument."""

        # VisaInstrument close
        if getattr(self, 'visa_handle', None):
            self.visa_handle.close()

        # Instrument close
        if hasattr(self, 'connection') and hasattr(self.connection, 'close'):
            self.connection.close()

        strip_attrs(self, whitelist=['name'])
        self.remove_instance(self)



class AMI430_VISA(AMI430, IPToVisa):
    pass
