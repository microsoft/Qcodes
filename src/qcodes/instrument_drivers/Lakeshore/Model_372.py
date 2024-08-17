"""
This contains an alias of the Lakeshore Model 372 driver.
It will eventually be deprecated and removed
"""

from typing import TYPE_CHECKING, Any, ClassVar

import qcodes.validators as vals
from qcodes.instrument_drivers.Lakeshore.lakeshore_base import (
    BaseOutput,  # pyright: ignore
    BaseSensorChannel,  # pyright: ignore
    LakeshoreBase,
)
from qcodes.parameters import Group, GroupParameter

from .Lakeshore_model_372 import LakeshoreModel372Channel as Model_372_Channel
from .Lakeshore_model_372 import LakeshoreModel372Output as Output_372

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from qcodes.instrument import VisaInstrumentKWArgs

# There are 16 sensors channels (a.k.a. measurement inputs) in Model 372
_n_channels = 16


class Model_372(LakeshoreBase):
    """
    Lakeshore Model 372 Temperature Controller Driver

    Note that interaction with the control input (referred to as 'A' in the
    Computer Interface Operation section of the manual) is not implemented.
    """

    channel_name_command: ClassVar[dict[str, str]] = {
        f"ch{i:02}": str(i) for i in range(1, 1 + _n_channels)
    }
    input_channel_parameter_values_to_channel_name_on_instrument: ClassVar[
        dict[int, str]
    ] = {i: f"ch{i:02}" for i in range(1, 1 + _n_channels)}

    CHANNEL_CLASS = Model_372_Channel

    def __init__(
        self, name: str, address: str, **kwargs: "Unpack[VisaInstrumentKWArgs]"
    ) -> None:
        super().__init__(name, address, **kwargs)

        heaters = {"sample_heater": 0, "warmup_heater": 1, "analog_heater": 2}
        for heater_name, heater_index in heaters.items():
            self.add_submodule(heater_name, Output_372(self, heater_name, heater_index))  # type: ignore[arg-type]
