from __future__ import annotations

import sys
from typing import Any

if sys.version_info >= (3, 9):
    from importlib.resources import as_file, files
else:
    from importlib_resources import as_file, files

import qcodes.validators as vals
from qcodes.instrument_drivers.american_magnetics.AMI430 import AMI430
from qcodes.logger import get_instrument_logger
from qcodes.utils import strip_attrs

from .instrument import Instrument
from .ip import IPInstrument

# previous to introducing the `InstrumentLoggerAdapter` the IPToVisa instrument
# was logging in the name of the `VisaInstrument`. To maintain that behaviour
# import the `instrument.visa.log` and log to this one.
from .visa import VISA_LOGGER, VisaInstrument

# This module provides a class to make an IPInstrument behave like a
# VisaInstrument. This is only meant for use with the PyVISA-sim backend
# for testing purposes. If you really need an IPInstrument to become
# a VisaInstrument for actual instrument control, please rewrite the driver.

# At the end of the module, a 'zoo' of was-ip-is-now-visa drivers can be found.
# Such a driver is just a two-line class definition.


class IPToVisa(VisaInstrument, IPInstrument):  # type: ignore[misc]
    """
    Class to inject an VisaInstrument like behaviour in an
    IPInstrument that we'd like to use as a VISAInstrument with the
    simulation back-end.
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

    def __init__(
        self,
        name: str,
        address: str,
        port: int | None,
        pyvisa_sim_file: str,
        device_clear: bool = False,
        terminator: str = "\n",
        timeout: float = 3,
        **kwargs: Any,
    ):

        # remove IPInstrument-specific kwargs
        ipkwargs = ['write_confirmation']
        newkwargs = {kw: val for (kw, val) in kwargs.items()
                     if kw not in ipkwargs}

        Instrument.__init__(self, name, **newkwargs)
        self.visa_log = get_instrument_logger(self, VISA_LOGGER)

        ##################################################
        # __init__ of VisaInstrument

        self.add_parameter('timeout',
                           get_cmd=self._get_visa_timeout,
                           set_cmd=self._set_visa_timeout,
                           unit='s',
                           vals=vals.MultiType(vals.Numbers(min_value=0),
                                               vals.Enum(None)))

        traversable_handle = files("qcodes.instrument.sims") / pyvisa_sim_file
        with as_file(traversable_handle) as sim_visalib_path:
            self.visalib = f"{str(sim_visalib_path)}@sim"
            self.set_address(address=address)

        if device_clear:
            self.device_clear()

        self.set_terminator(terminator)
        self.timeout.set(timeout)

    def close(self) -> None:
        """Disconnect and irreversibly tear down the instrument."""

        # VisaInstrument close
        if getattr(self, 'visa_handle', None):
            self.visa_handle.close()

        # Instrument close
        if hasattr(self, 'connection') and hasattr(self.connection, 'close'):
            self.connection.close()

        strip_attrs(self, whitelist=['name'])
        self.remove_instance(self)


class AMI430_VISA(AMI430, IPToVisa):  # type: ignore[misc]
    pass
