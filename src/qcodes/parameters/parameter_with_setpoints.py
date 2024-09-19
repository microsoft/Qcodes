from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from qcodes.validators import Arrays, Validator

from .parameter import Parameter

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from .parameter_base import ParamDataType, ParameterBase

LOG = logging.getLogger(__name__)


class ParameterWithSetpoints(Parameter):
    """
    A parameter that has associated setpoints. The setpoints is nothing
    more than a list of other parameters that describe the values, names
    and units of the setpoint axis for this parameter.

    In most cases this will probably be a parameter that returns an array.
    It is expected that the setpoint arrays are 1D arrays such that the
    combined shape of the parameter e.g. if parameter is of shape (m,n)
    `setpoints` is a list of parameters of shape (m,) and (n,)

    In all other ways this is identical to  :class:`Parameter`. See the
    documentation of :class:`Parameter` for more details.
    """

    def __init__(
        self,
        name: str,
        *,
        vals: Validator[Any] | None = None,
        setpoints: Sequence[ParameterBase] | None = None,
        snapshot_get: bool = False,
        snapshot_value: bool = False,
        **kwargs: Any,
    ) -> None:
        if not isinstance(vals, Arrays):
            raise ValueError(
                f"A ParameterWithSetpoints must have an Arrays "
                f"validator got {type(vals)}"
            )
        if vals.shape_unevaluated is None:
            raise RuntimeError(
                "A ParameterWithSetpoints must have a shape "
                "defined for its validator."
            )

        super().__init__(
            name=name,
            vals=vals,
            snapshot_get=snapshot_get,
            snapshot_value=snapshot_value,
            **kwargs,
        )
        if setpoints is None:
            self.setpoints = []
        else:
            self.setpoints = setpoints

        self._validate_on_get = True

    @property
    def setpoints(self) -> Sequence[ParameterBase]:
        """
        Sequence of parameters to use as setpoints for this parameter.

        :getter: Returns a list of parameters currently used for setpoints.
        :setter: Sets the parameters to be used as setpoints from a sequence.
            The combined shape of the parameters supplied must be consistent
            with the data shape of the data returned from get on the parameter.
        """
        return self._setpoints

    @setpoints.setter
    def setpoints(self, setpoints: Sequence[ParameterBase]) -> None:
        for setpointarray in setpoints:
            if not isinstance(setpointarray, Parameter):
                raise TypeError(
                    f"Setpoints is of type {type(setpointarray)}"
                    f" expected a QCoDeS parameter"
                )
        self._setpoints = setpoints

    def validate_consistent_shape(self) -> None:
        """
        Verifies that the shape of the Array Validator of the parameter
        is consistent with the Validator of the Setpoints. This requires that
        both the setpoints and the actual parameters have validators
        of type Arrays with a defined shape.
        """

        if not isinstance(self.vals, Arrays):
            raise ValueError(
                f"Can only validate shapes for parameters "
                f"with Arrays validator. {self.name} does "
                f"not have an Arrays validator."
            )
        output_shape = self.vals.shape_unevaluated
        setpoints_shape_list: list[int | Callable[[], int] | None] = []
        for sp in self.setpoints:
            if not isinstance(sp.vals, Arrays):
                raise ValueError(
                    f"Can only validate shapes for parameters "
                    f"with Arrays validator. {sp.name} is "
                    f"a setpoint vector but does not have an "
                    f"Arrays validator"
                )
            if sp.vals.shape_unevaluated is not None:
                setpoints_shape_list.extend(sp.vals.shape_unevaluated)
            else:
                setpoints_shape_list.append(sp.vals.shape_unevaluated)
        setpoints_shape = tuple(setpoints_shape_list)

        if output_shape is None:
            raise ValueError(
                f"Trying to validate shape but parameter "
                f"{self.name} does not define a shape"
            )
        if None in output_shape or None in setpoints_shape:
            raise ValueError(
                f"One or more dimensions have unknown shape "
                f"when comparing output: {output_shape} to "
                f"setpoints: {setpoints_shape}"
            )

        if output_shape != setpoints_shape:
            raise ValueError(
                f"Shape of output is not consistent with "
                f"setpoints. Output is shape {output_shape} and "
                f"setpoints are shape {setpoints_shape}"
            )
        LOG.debug(
            f"For parameter {self.full_name} verified "
            f"that {output_shape} matches {setpoints_shape}"
        )

    def validate(self, value: ParamDataType) -> None:
        """
        Overwrites the standard ``validate`` method to also check the the
        parameter has consistent shape with its setpoints. This only makes
        sense if the parameter has an Arrays
        validator

        Arguments are passed to the super method
        """
        if isinstance(self.vals, Arrays):
            self.validate_consistent_shape()
        super().validate(value)


def expand_setpoints_helper(
    parameter: ParameterWithSetpoints, results: ParamDataType | None = None
) -> list[tuple[ParameterBase, ParamDataType]]:
    """
    A helper function that takes a :class:`.ParameterWithSetpoints` and
    acquires the parameter along with it's setpoints. The data is returned
    in a format prepared to insert into the dataset.

    Args:
        parameter: A :class:`.ParameterWithSetpoints` to be acquired and
            expanded
        results: The data for the given parameter. Typically the output of
            `parameter.get()`. If None this function will call `parameter.get`

    Returns:
        A list of tuples of parameters and values for the specified parameter
        and its setpoints.
    """
    if not isinstance(parameter, ParameterWithSetpoints):
        raise TypeError(
            f"Expanding setpoints only works for ParameterWithSetpoints. "
            f"Supplied a {type(parameter)}"
        )
    res = []
    setpoint_params = []
    setpoint_data = []
    for setpointparam in parameter.setpoints:
        these_setpoints = setpointparam.get()
        setpoint_params.append(setpointparam)
        setpoint_data.append(these_setpoints)
    output_grids = np.meshgrid(*setpoint_data, indexing="ij")
    for param, grid in zip(setpoint_params, output_grids):
        res.append((param, grid))
    if results is None:
        data = parameter.get()
    else:
        data = results
    res.append((parameter, data))
    return res
