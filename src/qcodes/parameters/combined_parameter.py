from __future__ import annotations

import collections
import collections.abc
import logging
from copy import copy
from typing import TYPE_CHECKING, Any

import numpy as np

from qcodes.metadatable import Metadatable
from qcodes.utils import full_class

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from .parameter import Parameter

_LOG = logging.getLogger(__name__)


def combine(
    *parameters: Parameter,
    name: str,
    label: str | None = None,
    unit: str | None = None,
    units: str | None = None,
    aggregator: Callable[..., Any] | None = None,
) -> CombinedParameter:
    """
    Combine parameters into one sweepable parameter

    A combined parameter sets all the combined parameters at every point
    of the sweep. The sets are called in the same order the parameters are,
    and sequentially.

    Args:
        *parameters: The parameters to combine.
        name: The name of the paramter.
        label: The label of the combined parameter.
        unit: The unit of the combined parameter.
        units: Deprecated argument left for backwards compatibility. Do not use.
        aggregator: A function to aggregate
            the set values into one.
    """
    my_parameters = list(parameters)
    multi_par = CombinedParameter(my_parameters, name, label, unit, units, aggregator)
    return multi_par


class CombinedParameter(Metadatable):
    """
    A combined parameter. It sets all the combined parameters at every
    point of the sweep. The sets are called in the same order
    the parameters are, and sequentially.

    Args:
        *parameters: The parameters to combine.
        name: The name of the parameter
        label: The label of the combined parameter
        unit: The unit of the combined parameter
        units: Deprecated argument left for backwards compatibility. Do not use.
        aggregator: A function to aggregate the set values into one
    """

    def __init__(
        self,
        parameters: Sequence[Parameter],
        name: str,
        label: str | None = None,
        unit: str | None = None,
        units: str | None = None,
        aggregator: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__()
        # TODO(giulioungaretti)temporary hack
        # starthack
        # this is a dummy parameter
        # that mimicks the api that a normal parameter has
        if not name.isidentifier():
            raise ValueError(
                f"Parameter name must be a valid identifier "
                f"got {name} which is not. Parameter names "
                f"cannot start with a number and "
                f"must not contain spaces or special characters"
            )

        self.parameter = lambda: None
        # mypy will complain that a callable does not have these attributes
        # but you can still create them here.
        self.parameter.full_name = name  # type: ignore[attr-defined]
        self.parameter.name = name  # type: ignore[attr-defined]
        self.parameter.label = label  # type: ignore[attr-defined]

        if units is not None:
            _LOG.warning(
                f"`units` is deprecated for the "
                f"`CombinedParameter` class, use `unit` instead. {self!r}"
            )
            if unit is None:
                unit = units
        self.parameter.unit = unit  # type: ignore[attr-defined]
        self.setpoints: list[Any] = []
        # endhack
        self.parameters = parameters
        self.sets = [parameter.set for parameter in self.parameters]
        self.dimensionality = len(self.sets)

        if aggregator:
            self.f = aggregator
            setattr(self, "aggregate", self._aggregate)

    def set(self, index: int) -> list[Any]:
        """
        Set multiple parameters.

        Args:
            index: the index of the setpoints one wants to set

        Returns:
            list of values that where actually set
        """
        values = self.setpoints[index]
        for setFunction, value in zip(self.sets, values):
            setFunction(value)
        return values

    def sweep(self, *array: np.ndarray) -> CombinedParameter:
        """
        Creates a new combined parameter to be iterated over.
        One can sweep over either:

         - n array of length m
         - one nxm array

        where n is the number of combined parameters
        and m is the number of setpoints

        Args:
            *array: Array(s) of setpoints.

        Returns:
            combined parameter
        """
        # if it's a list of arrays, convert to one array
        if len(array) > 1:
            dim = {len(a) for a in array}
            if len(dim) != 1:
                raise ValueError("Arrays have different number of setpoints")
            nparray = np.array(array).transpose()
        elif len(array) == 1:
            # cast to array in case users
            # decide to not read docstring
            # and pass a 2d list
            nparray = np.array(array[0])
        else:
            raise ValueError("Need at least one array to sweep over.")
        new = copy(self)
        _error_msg = """ Dimensionality of array does not match\
                        the number of parameter combined. Expected a \
                        {} dimensional array, got a {} dimensional array. \
                        """
        try:
            if nparray.shape[1] != self.dimensionality:
                raise ValueError(
                    _error_msg.format(self.dimensionality, nparray.shape[1])
                )
        except KeyError:
            # this means the array is 1d
            raise ValueError(_error_msg.format(self.dimensionality, 1))

        new.setpoints = nparray.tolist()
        return new

    def _aggregate(self, *vals: Any) -> Any:
        # check f args
        return self.f(*vals)

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.setpoints)))

    def __len__(self) -> int:
        # dimension of the sweep_values
        # i.e. how many setpoint
        return np.shape(self.setpoints)[0]

    def snapshot_base(
        self,
        update: bool | None = False,
        params_to_skip_update: Sequence[str] | None = None,
    ) -> dict[Any, Any]:
        """
        State of the combined parameter as a JSON-compatible dict (everything
        that the custom JSON encoder class
        :class:`.NumpyJSONEncoder` supports).

        Args:
            update: ``True`` or ``False``.
            params_to_skip_update: Unused in this subclass.

        Returns:
            dict: Base snapshot.
        """
        meta_data: dict[str, Any] = collections.OrderedDict()
        meta_data["__class__"] = full_class(self)
        param = self.parameter
        meta_data["unit"] = param.unit  # type: ignore[attr-defined]
        meta_data["label"] = param.label  # type: ignore[attr-defined]
        meta_data["full_name"] = param.full_name  # type: ignore[attr-defined]
        meta_data["aggregator"] = repr(getattr(self, "f", None))
        for parameter in self.parameters:
            meta_data[str(parameter)] = parameter.snapshot()

        return meta_data
