from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from .multi_parameter import MultiParameter

if TYPE_CHECKING:
    from collections.abc import Sequence

    from qcodes.instrument.channel import InstrumentModule

    from .parameter_base import ParamRawDataType

InstrumentModuleType = TypeVar("InstrumentModuleType", bound="InstrumentModule")
_LOG = logging.getLogger(__name__)


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

    def set_raw(self, value: ParamRawDataType | Sequence[ParamRawDataType]) -> None:
        """
        Set all parameters to this/these value(s).

        Args:
            value: The value(s) to set to. The type is given by the
                underlying parameter.
        """
        try:
            for chan in self._channels:
                getattr(chan, self._param_name).set(value)
        except Exception as err:
            try:
                # Catch wrong length of value before any setting is done
                value_list = list(value)
                if len(value_list) != len(self._channels):
                    raise ValueError
                for chan, val in zip(self._channels, value_list):
                    getattr(chan, self._param_name).set(val)
            except (TypeError, ValueError):
                note = (
                    "Value should either be valid for a single parameter of the channel list "
                    "or a sequence of valid values of the same length as the list."
                )
                if sys.version_info >= (3, 11):
                    err.add_note(note)
                else:
                    _LOG.error(note)
                raise err from None

    @property
    def full_names(self) -> tuple[str, ...]:
        """
        Overwrite full_names because the instrument name is already included
        in the name. This happens because the instrument name is included in
        the channel name merged into the parameter name above.
        """

        return self.names
