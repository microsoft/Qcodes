"""
This contains an alias of the Lakeshore Model 336 driver.
It will eventually be deprecated and removed
"""

from typing import TYPE_CHECKING, Any, ClassVar

import qcodes.validators as vals
from qcodes.parameters import Group, GroupParameter

from .lakeshore_base import (
    BaseOutput,  # pyright: ignore
    BaseSensorChannel,  # pyright: ignore
    LakeshoreBase,
)
from .Lakeshore_model_336 import LakeshoreModel336Channel as Model_336_Channel
from .Lakeshore_model_336 import (
    LakeshoreModel336CurrentSource as Output_336_CurrentSource,
)
from .Lakeshore_model_336 import (
    LakeshoreModel336VoltageSource as Output_336_VoltageSource,
)
from .Lakeshore_model_336 import _channel_name_to_command_map

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from qcodes.instrument import VisaInstrumentKWArgs


class Model_336(LakeshoreBase):
    """
    Lakeshore Model 336 Temperature Controller Driver
    """

    channel_name_command: ClassVar[dict[str, str]] = _channel_name_to_command_map

    CHANNEL_CLASS = Model_336_Channel

    input_channel_parameter_values_to_channel_name_on_instrument = \
        _channel_name_to_command_map

    def __init__(
        self, name: str, address: str, **kwargs: "Unpack[VisaInstrumentKWArgs]"
    ) -> None:
        super().__init__(name, address, **kwargs)

        self.output_1 = Output_336_CurrentSource(self, "output_1", 1)  # type: ignore[arg-type]
        self.output_2 = Output_336_CurrentSource(self, "output_2", 2)  # type: ignore[arg-type]
        self.output_3 = Output_336_VoltageSource(self, "output_3", 3)  # type: ignore[arg-type]
        self.output_4 = Output_336_VoltageSource(self, "output_4", 4)  # type: ignore[arg-type]
