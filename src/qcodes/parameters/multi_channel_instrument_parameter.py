from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from .multi_parameter import MultiParameter
from .parameter_base import ParamRawDataType

if TYPE_CHECKING:
    from qcodes.instrument.channel import InstrumentModule

InstrumentModuleType = TypeVar("InstrumentModuleType", bound="InstrumentModule")


class MultiChannelInstrumentParameter(MultiParameter, Generic[InstrumentModuleType]):
    """
    Parameter to get or set multiple channels simultaneously.

    Will normally be created by a :class:`ChannelList` and not directly by
    anything else.

    Args:
        channels: A list of channels which we can operate on
          simultaneously.
        param_name: Name of the multichannel parameter
    """

    def __init__(
        self,
        channels: Sequence[InstrumentModuleType],
        param_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._channels = channels
        self._param_name = param_name

    def get_raw(self) -> tuple[ParamRawDataType, ...]:
        """
        Return a tuple containing the data from each of the channels in the
        list.
        """
        return tuple(chan.parameters[self._param_name].get() for chan in self._channels)

    def set_raw(self, value: ParamRawDataType) -> None:
        """
        Set all parameters to this value.

        Args:
            value: The value to set to. The type is given by the
                underlying parameter.
        """
        for chan in self._channels:
            getattr(chan, self._param_name).set(value)

    @property
    def full_names(self) -> tuple[str, ...]:
        """
        Overwrite full_names because the instrument name is already included
        in the name. This happens because the instrument name is included in
        the channel name merged into the parameter name above.
        """

        return self.names
