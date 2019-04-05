import logging
from typing import Union
import visa

from qcodes import VisaInstrument
from qcodes.instrument_drivers.tektronix.Keithley_2400 import Keithley_2400


log = logging.getLogger(__name__)


class _Keithley_2450(VisaInstrument):
    """
    A 2450 driver with the full set of features. Note that the command set
    between the 2400 and 2450 are different (when not running in compatibility mode)
    and its not just a question of adjusting ranges or other minor modifications.
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
    instrument to SCPI or SCPI2400 instead of the Tektronix propriety TSP
    language, even before initializing the parent Keithley_2400 class. Otherwise
    any SCPI command in the init of that class will likely make this driver break.
    """

    def __new__(
            cls,
            name: str,
            address: str,
            compatibility_mode: Union[None, bool] = None,
            **kwargs
    ) -> VisaInstrument:
        """
        Args:
            name
            address
            compatibility_mode: If None, the current compatibility mode setting will be
                unchanged.

        Returns:
            VisaInstrument
        """

        resource_manager = visa.ResourceManager()
        raw_instrument = resource_manager.open_resource(address)
        current_language = raw_instrument.query("*LANG?").strip()

        if current_language == "TSP":
            log.warning("The instrument is in TSP mode which is not supported.")
            if compatibility_mode is None:
                log.warning("Switching 'compatibility mode' to 'True'")
                compatibility_mode = True

        if compatibility_mode is None:
            target_language = current_language
        elif compatibility_mode:
            target_language = "SCPI2400"
        else:
            target_language = "SCPI"

        if current_language != target_language:

            raw_instrument.write(f"*LANG {target_language}")

            raise RuntimeError(
                f"A language change from {current_language} to {target_language} "
                f"is needed. This has been adjusted, but a power cycle is "
                f"needed to make this take effect. Please reboot the instrument "
                f"and try again"
            )

        if current_language == "SCPI2400":
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
