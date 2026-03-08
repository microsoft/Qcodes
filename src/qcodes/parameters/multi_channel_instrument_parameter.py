from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar

from qcodes.utils import QCoDeSDeprecationWarning

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

    _DEPRECATED_POSITIONAL_ARGS: ClassVar[tuple[str, ...]] = (
        "channels",
        "param_name",
    )

    _CHANNELS_UNSET: Any = object()
    _PARAM_NAME_UNSET: Any = object()

    def __init__(
        self,
        *args: Any,
        channels: Sequence[InstrumentModuleType] = _CHANNELS_UNSET,
        param_name: str = _PARAM_NAME_UNSET,
        **kwargs: Any,
    ) -> None:
        if args:
            # TODO: After QCoDeS 0.57 remove the args argument and delete this code block.
            positional_names = self._DEPRECATED_POSITIONAL_ARGS
            if len(args) > len(positional_names):
                raise TypeError(
                    f"{type(self).__name__}.__init__() takes at most "
                    f"{len(positional_names) + 1} positional arguments "
                    f"({len(args) + 1} given)"
                )

            _defaults: dict[str, Any] = {
                "channels": self._CHANNELS_UNSET,
                "param_name": self._PARAM_NAME_UNSET,
            }

            _kwarg_vals: dict[str, Any] = {
                "channels": channels,
                "param_name": param_name,
            }

            for i in range(len(args)):
                arg_name = positional_names[i]
                if _kwarg_vals[arg_name] is not _defaults[arg_name]:
                    raise TypeError(
                        f"{type(self).__name__}.__init__() got multiple "
                        f"values for argument '{arg_name}'"
                    )

            positional_arg_names = positional_names[: len(args)]
            names_str = ", ".join(f"'{n}'" for n in positional_arg_names)
            warnings.warn(
                f"Passing {names_str} as positional argument(s) to "
                f"{type(self).__name__} is deprecated. "
                f"Please pass them as keyword arguments.",
                QCoDeSDeprecationWarning,
                stacklevel=2,
            )

            _pos = dict(zip(positional_names, args))
            channels = _pos.get("channels", channels)
            param_name = _pos.get("param_name", param_name)

        if channels is self._CHANNELS_UNSET:
            raise TypeError(
                f"{type(self).__name__}.__init__() missing required "
                f"keyword argument: 'channels'"
            )
        if param_name is self._PARAM_NAME_UNSET:
            raise TypeError(
                f"{type(self).__name__}.__init__() missing required "
                f"keyword argument: 'param_name'"
            )

        super().__init__(**kwargs)
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

                err.add_note(note)

                raise err from None

    @property
    def full_names(self) -> tuple[str, ...]:
        """
        Overwrite full_names because the instrument name is already included
        in the name. This happens because the instrument name is included in
        the channel name merged into the parameter name above.
        """

        return self.names
