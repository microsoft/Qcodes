from __future__ import annotations

import logging
from collections import OrderedDict, namedtuple
from typing import TYPE_CHECKING, Any

from .delegate_parameter import DelegateParameter
from .group_parameter import Group, GroupParameter
from .parameter_base import ParamDataType, ParameterBase, ParamRawDataType

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from qcodes.instrument.base import InstrumentBase

    from .parameter import Parameter


_log = logging.getLogger(__name__)


class DelegateGroupParameter(DelegateParameter, GroupParameter):
    def __init__(
        self,
        name: str,
        source: Parameter | None,
        instrument: InstrumentBase | None = None,
        initial_value: float | str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            source=source,
            instrument=instrument,
            initial_value=initial_value,
            **kwargs,
        )


class DelegateGroup(Group):
    """The DelegateGroup combines :class:`.DelegateParameter` s that
    are to be gotten or set using one :class:`.GroupedParameter`.
    Each :class:`.DelegateParameter` maps to one source parameter
    that is individually set or gotten on an instrument. These
    parameters can originate from the same or different instruments.

    The class :class:`.DelegateGroup` is used within the
    :class:`GroupedParameter` class in order to get and set the
    :class:`.DelegateParameter` s either via their default get and set
    methods or via a custom get or set method.

    The value to be set can be passed to the set method either via a
    dictionary, where the keys are the names of the
    :class:`.DelegateParameter` s contained in the :class:`DelegateGroup`,
    or a single value, if a custom setter is defined or if the group
    only contains a single :class:`.DelegateParameter`.

    The value returned by the get method is passed through a formatter.
    By default, the formatter returns the :class:`.DelegateParameter`
    values in a namedtuple, where the keys are the names of the
    :class:`.DelegateParameter` s. In the special case where the
    :class:`.DelegateGroup` only contains one :class:`.DelegateParameter`,
    the formatter simply returns the individual value. Optionally,
    the formatter can be customized and specified via the constructor.
    The formatter takes as input the values of the :class:`.DelegateParameter` s
    as positional arguments in the order at which the
    :class:`.DelegateParameter` s are specified.

    Args:
        name: Name of the DelegateGroup
        parameters: DelegateParameters to group together
        parameter_names: Optional names of parameters, defaults to the
            parameter `name` attributes
        setter: Optional function to call for setting the grouped parameters,
            should take one argument `value`. Defaults to set_parameters(),
            which sets each parameter using its .set() method.
        getter: Optional function to call for getting the grouped parameters.
            Defaults to .get_parameters(), which runs the get() method for each
            parameter.
        formatter: Optional formatter for value returned by get_parameters(),
            defaults to a namedtuple with the parameter names as keys.
    """

    def __init__(
        self,
        name: str,
        parameters: Sequence[DelegateGroupParameter],
        parameter_names: Iterable[str] | None = None,
        setter: Callable[..., Any] | None = None,
        getter: Callable[..., Any] | None = None,
        formatter: Callable[..., Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(parameters=parameters, single_instrument=False, **kwargs)

        self.name = name
        self._parameter_names = parameter_names or [_e.name for _e in parameters]
        self._set_fn = setter
        self._get_fn = getter
        self._params = parameters
        self._parameters = OrderedDict(zip(self._parameter_names, parameters))

        if formatter is None and len(parameters) == 1:
            self._formatter = lambda result: result
        elif formatter is None:
            self._formatter = self._namedtuple
        else:
            self._formatter = formatter

    def _namedtuple(self, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        return namedtuple(self.name, self._parameter_names)(*args, **kwargs)

    def set(self, value: ParamDataType | Mapping[str, ParamDataType]) -> None:
        if self._set_fn is not None:
            self._set_fn(value)
        else:
            if not isinstance(value, dict):
                value = {name: value for name in self._parameter_names}
            self.set_parameters(value)

    def get(self) -> Any:
        if self._get_fn is not None:
            return self._get_fn()
        else:
            return self.get_parameters()

    def get_parameters(self) -> Any:
        return self._formatter(*(_p.get() for _p in self.parameters.values()))

    def _set_from_dict(self, calling_dict: Mapping[str, ParamRawDataType]) -> None:
        for name, p in list(self.parameters.items()):
            p.set(calling_dict[name])

    @property
    def source_parameters(self) -> tuple[Parameter | None, ...]:
        """Get source parameters of each DelegateParameter"""
        return tuple(p.source for p in self._params)


class GroupedParameter(ParameterBase):
    """
    The GroupedParameter wraps one or more :class:`.DelegateParameter` s,
    such that those parameters can be accessed as if they were one
    parameter.

    The :class:`GroupedParameter` uses a :class:`DelegateGroup` to keep
    track of the :class:`.DelegateParameter` s. Mainly, this class is a
    thin wrapper around the :class:`DelegateGroup`, and mainly exists
    in order to allow for it to be used as a :class:`ParameterBase`.

    This class can be seen as the opposite of a :class:`GroupParameter`,
    which is a class to create parameters that are set with a single get
    and set string command on *the same* instrument but need to be accessed
    separately. In contrast, the :class:`GroupedParameter` allows grouped
    access to parameters that are normally separate, and can be associated
    with *different* instruments.

    Args:
        name: Grouped parameter name.
        group: Group that contains the target parameter(s).
        unit: The unit of measure. Use ``''`` for unitless.
        label: Optional label, defaults to parameter name.
        default set method(s).
    """

    def __init__(
        self,
        name: str,
        group: DelegateGroup,
        unit: str | None = None,
        label: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(name, **kwargs)
        self.label = name if label is None else label
        self.unit = unit if unit is not None else ""
        self._group = group

    @property
    def group(self) -> DelegateGroup:
        """
        The group that contains the target parameters.
        """
        return self._group

    @property
    def parameters(self) -> dict[str, GroupParameter]:
        """Get delegate parameters wrapped by this GroupedParameter"""
        return self.group.parameters

    @property
    def source_parameters(self) -> tuple[Parameter | None, ...]:
        """Get source parameters of each DelegateParameter"""
        return self.group.source_parameters

    def get_raw(self) -> ParamDataType | Mapping[str, ParamDataType]:
        """Get parameter raw value"""
        return self.group.get_parameters()

    def set_raw(self, value: ParamDataType | Mapping[str, ParamDataType]) -> None:
        """Set parameter raw value

        Args:
            value: Parameter value to set

        Returns:
            float: Returns the parameter value
        """
        self.group.set(value)

    def __repr__(self) -> str:
        output = f"GroupedParameter(name={self.name}"
        if self.source_parameters:
            source_parameters = ", ".join(str(_) for _ in self.source_parameters)
            output += f", source_parameters=({source_parameters})"
        output += ")"
        return output
