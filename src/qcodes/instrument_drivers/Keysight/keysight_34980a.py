import logging
import re
import warnings
from functools import wraps
from typing import TYPE_CHECKING, TypeVar

from typing_extensions import ParamSpec

from qcodes import validators as vals
from qcodes.instrument import VisaInstrument, VisaInstrumentKWArgs

from .keysight_34934a import Keysight34934A
from .keysight_34980a_submodules import Keysight34980ASwitchMatrixSubModule

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Concatenate

    from typing_extensions import Unpack

KEYSIGHT_MODELS = {"34934A": Keysight34934A}

S = TypeVar("S", bound="Keysight34980A")
P = ParamSpec("P")
T = TypeVar("T")


def post_execution_status_poll(
    func: "Callable[Concatenate[S, P], T]",
) -> "Callable[Concatenate[S, P], T]":
    """
    Generates a decorator that clears the instrument's status registers
    before executing the actual call and reads the status register after the
    function call to determine whether an error occurs.

    Args:
        func: function to wrap
    """

    @wraps(func)
    def wrapper(self: S, *args: P.args, **kwargs: P.kwargs) -> T:
        self.clear_status()
        retval = func(self, *args, **kwargs)

        stb = self.get_status()
        if stb:
            warnings.warn(
                f"Instrument status byte indicates an error occurred "
                f"(value of STB was: {stb})! Use `get_error` method "
                f"to poll error message.",
                stacklevel=2,
            )
        return retval

    return wrapper


class Keysight34980A(VisaInstrument):
    """
    QCodes driver for 34980A switch/measure unit
    """

    default_terminator = "\n"

    def __init__(
        self, name: str, address: str, **kwargs: "Unpack[VisaInstrumentKWArgs]"
    ):
        """
        Create an instance of the instrument.

        Args:
            name: Name of the instrument instance
            address: Visa-resolvable instrument address.
            **kwargs: kwargs are forwarded to base class.
        """
        super().__init__(name, address, **kwargs)

        self._total_slot = 8
        self._system_slots_info_dict: dict[int, dict[str, str]] | None = None
        self.module = dict.fromkeys(self.system_slots_info.keys())
        self.scan_slots()
        self.connect_message()

    def get_status(self) -> int:
        """
        Queries status register

        Returns:
            0 if there is no error
        """
        msg = super().ask("*ESR?")
        nums = list(map(int, re.findall(r"\d+", msg)))
        return nums[0]

    def get_error(self) -> str:
        """
        Queries error queue

        Returns:
            error message, or '+0,"No error"' if there is no error
        """
        msg = super().ask(":SYST:ERR?")
        return msg

    def clear_status(self) -> None:
        """
        Clears status register and error queue of the instrument.
        """
        super().write("*CLS")

    def reset(self) -> None:
        """
        Performs an instrument reset.
        Does not reset error queue!
        """
        super().write("*RST")

    @post_execution_status_poll
    def ask(self, cmd: str) -> str:
        return super().ask(cmd)

    @post_execution_status_poll
    def write(self, cmd: str) -> None:
        return super().write(cmd)

    def scan_slots(self) -> None:
        """
        Scan the occupied slots and make an object for each switch matrix
        module installed
        """
        for slot in self.system_slots_info.keys():
            model_string = self.system_slots_info[slot]["model"]
            for model in KEYSIGHT_MODELS:
                if model in model_string:
                    sub_module_name = f"slot_{slot}_{model}"
                    sub_module = KEYSIGHT_MODELS[model](self, sub_module_name, slot)
                    self.module[slot] = sub_module
                    self.add_submodule(sub_module_name, sub_module)
                    break
            if self.module[slot] is None:
                sub_module_name = f"slot_{slot}_{model_string}_no_driver"
                sub_module_no_driver = Keysight34980ASwitchMatrixSubModule(
                    self, sub_module_name, slot
                )
                self.module[slot] = sub_module_no_driver
                self.add_submodule(sub_module_name, sub_module_no_driver)
                logging.warning(
                    f"can not find driver for {model_string} in slot {slot}"
                )

    @property
    def system_slots_info(self) -> dict[int, dict[str, str]]:
        if self._system_slots_info_dict is None:
            self._system_slots_info_dict = self._system_slots_info()
        return self._system_slots_info_dict

    def _system_slots_info(self) -> dict[int, dict[str, str]]:
        """
        the command SYST:CTYP? returns the following:
        Agilent Technologies,<Model Number>,<Serial Number>,<Firmware Rev>
        where <Model Number> is '0' if there is no module connected to the
        given slot

        Returns:
            a dictionary, with slot numbers as the keys, and vendor/model/
            serial/firmware dictionaries as the values
        """
        slots_dict = {}
        keys = ["vendor", "model", "serial", "firmware"]
        for i in range(1, self._total_slot + 1):
            identity = self.ask(f"SYST:CTYP? {i}").strip('"').split(",")
            if identity[1] != "0":
                slots_dict[i] = dict(zip(keys, identity))
        return slots_dict

    def disconnect_all(self, slot: int | None = None) -> None:
        """
        to open/disconnect all connections on select module

        Args:
            slot: slot number, between 1 and 8 (self._total_slot),
                    default value is None, which means all slots
        """
        cmd = "ROUT:OPEN:ALL"
        if slot is None:
            self.write(cmd)
        else:
            vals.Ints(min_value=1, max_value=self._total_slot).validate(slot)
            self.write(f"ROUT:OPEN:ALL {slot}")
