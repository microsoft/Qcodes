from qcodes.instrument.base import Instrument
from qcodes.instrument.ip import IPInstrument
from qcodes.instrument.visa import VisaInstrument
from qcodes.instrument_drivers.american_magnetics.AMI430 import AMI430
from qcodes.utils.helpers import strip_attrs
import qcodes.utils.validators as vals

# This module provides a class to make an IPInstrument behave like a
# VisaInstrument. This is only meant for use with the PyVISA-sim backend
# for testing purposes. If you really need an IPInstrument to become
# a VisaInstrument for actual instrument control, please rewrite the driver.

# At the end of the module, a 'zoo' of was-ip-is-now-visa drivers can be found.
# Such a driver is just a two-line class definition.


class IPToVisa(VisaInstrument, IPInstrument): # type: ignore
    """
    Class to inject an VisaInstrument like behaviour in an
    IPInstrument that we'd like to use as a VISAInstrument with the
    simulation backend.
    The idea is to inject this class just before the IPInstrument in
    the MRO. To avoid IPInstrument to ever take any effect, we sidestep
    it during the __init__ by calling directly to Instrument (which then
    class up through the chain) and explicitly reimplementing the __init__
    of VisaInstrument. We also must reimplement close, as that method
    calls super.

    Only meant for testing/simulation purposes!
    Do not use this for actual instrument control, there could be many
    nasty surprises.
    """

    def __init__(self, name, address, port, visalib,
                 metadata=None, device_clear=False, terminator='\n',
                 timeout=3, **kwargs):

        # remove IPInstrument-specific kwargs
        ipkwargs = ['write_confirmation']
        newkwargs = {kw: val for (kw, val) in kwargs.items()
                     if kw not in ipkwargs}

        Instrument.__init__(self, name, metadata=metadata, **newkwargs)

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


class AMI430_VISA(AMI430, IPToVisa): # type: ignore
    pass
