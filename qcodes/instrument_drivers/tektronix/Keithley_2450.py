import visa

from qcodes import VisaInstrument
from qcodes.instrument_drivers.tektronix.Keithley_2400 import Keithley_2400


class _Keithley_2450(VisaInstrument):
    """
    A 2450 driver with the full set of features
    """

    def __init__(self, name, address, **kwargs):
        super().__init__(name, address, **kwargs)
        raise NotImplementedError("This driver has not been developed yet")


class Keithley_2450:
    """
    The driver for the Keithley 2450 source meter. We can select the command
    set of this instrument to be compatible with the 2400, but this will reduce
    the feature set. We will accept this for now.

    Note that before doing anything else, we need to set the language of the
    instrument to SCPI instead of the Tektronix propriety TSP language, even
    before initializing the parent Keithley_2400 class. Otherwise any SCPI
    command in the init of that class will likely make this driver break.
    """

    def __new__(cls, name, address, compatibility_mode=True, **kwargs):

        resource_manager = visa.ResourceManager()
        raw_instrument = resource_manager.open_resource(address)
        current_language = raw_instrument.query("*LANG?").strip()

        if compatibility_mode:
            target_language = "SCPI2400"
        else:
            target_language = "SCPI"

        raw_instrument.write(f"*LANG {target_language}")

        if current_language == "TSP":
            raise RuntimeError(
                "This driver is not compatible with the TSP langauge. "
                "This has been adjusted to SCPI, but a power cycle is "
                "needed to make this take effect. Please reboot the instrument "
                "and try again"
            )

        if compatibility_mode:
            driver_instance = Keithley_2400(name, address, **kwargs)

            driver_instance.log.warning(
                "This driver is running in compatibility mode with the Keithley 2400. "
                "This means that a number of newer features introduced in the 2450 will "
                "not be available. Please consult Appendix D of the 2450 manual to see if "
                "this is acceptable to you"
            )

        else:
            driver_instance = _Keithley_2450(name, address, **kwargs)

        return driver_instance
