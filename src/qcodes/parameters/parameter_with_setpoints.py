from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Generic, Literal, NotRequired, TypedDict

import numpy as np

from qcodes.parameters.parameter import (
    Parameter,
    ParameterKWArgs,
)
from qcodes.parameters.parameter_base import (
    InstrumentTypeVar_co,
    ParameterBase,
    ParameterDataTypeVar,
    ParameterSet,
)
from qcodes.validators import Arrays

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from typing_extensions import Unpack

    from qcodes.dataset.data_set_protocol import ValuesType
    from qcodes.parameters.parameter_base import ParamDataType
    from qcodes.validators import Validator

LOG = logging.getLogger(__name__)


class ParameterWithSetpointsKWArgs(
    TypedDict,
    Generic[ParameterDataTypeVar, InstrumentTypeVar_co],
):
    """
    This TypedDict defines the type of the kwargs that can be passed to
    the ``ParameterWithSetpoints`` class.

    A subclass of ``ParameterWithSetpoints`` should take
    ``**kwargs: Unpack[ParameterWithSetpointsKWArgs]`` as input and forward
    this to the super class to ensure that it can accept all the arguments
    defined here.
    """

    # Members from ParameterBaseKWArgs are redeclared here
    # so that Sphinx can discover and document them.
    instrument: NotRequired[InstrumentTypeVar_co]
    """
    The instrument this parameter belongs to, if any.
    """
    snapshot_get: NotRequired[bool]
    """
    False prevents any update to the parameter during a snapshot,
    even if the snapshot was called with ``update=True``.
    Default True.
    """
    metadata: NotRequired[Mapping[Any, Any] | None]
    """
    Additional static metadata to add to this
    parameter's JSON snapshot.
    """
    step: NotRequired[float | None]
    """
    Max increment of parameter value.
    Larger changes are broken into multiple steps this size.
    When combined with delays, this acts as a ramp.
    """
    scale: NotRequired[float | Iterable[float] | None]
    """
    Scale to multiply value with before performing set.
    The internally multiplied value is stored in
    ``cache.raw_value``. Can account for a voltage divider.
    """
    offset: NotRequired[float | Iterable[float] | None]
    """
    Compensate for a parameter specific offset.
    get value = raw value - offset.
    set value = argument + offset.
    """
    inter_delay: NotRequired[float]
    """
    Minimum time (in seconds) between successive sets.
    If the previous set was less than this, it will wait until the
    condition is met. Can be set to 0 to go maximum speed with
    no errors.
    """
    post_delay: NotRequired[float]
    """
    Time (in seconds) to wait after the *start* of each set,
    whether part of a sweep or not. Can be set to 0 to go maximum
    speed with no errors.
    """
    val_mapping: NotRequired[Mapping[Any, Any] | None]
    """
    A bidirectional map of data/readable values to instrument codes,
    expressed as a dict: ``{data_val: instrument_code}``.
    """
    get_parser: NotRequired[Callable[..., Any] | None]
    """
    Function to transform the response from get to the final
    output value. See also ``val_mapping``.
    """
    set_parser: NotRequired[Callable[..., Any] | None]
    """
    Function to transform the input set value to an encoded
    value sent to the instrument. See also ``val_mapping``.
    """
    snapshot_value: NotRequired[bool]
    """
    False prevents parameter value to be stored in the snapshot.
    Useful if the value is large. Default True.
    """
    snapshot_exclude: NotRequired[bool]
    """
    True prevents parameter to be included in the snapshot.
    Useful if there are many of the same parameter which are
    clogging up the snapshot. Default False.
    """
    max_val_age: NotRequired[float | None]
    """
    The max time (in seconds) to trust a saved value obtained
    from ``cache.get`` (or ``get_latest``). If this parameter has not
    been set or measured more recently than this, perform an
    additional measurement.
    """
    vals: NotRequired[Validator[Any] | None]
    """
    A Validator object for this parameter.
    """
    abstract: NotRequired[bool | None]
    """
    Specifies if this parameter is abstract or not. Default is False.
    If the parameter is 'abstract', it *must* be overridden by a
    non-abstract parameter before the instrument containing this
    parameter can be instantiated.
    """
    bind_to_instrument: NotRequired[bool]
    """
    Should the parameter be registered as a delegate attribute
    on the instrument passed via the instrument argument.
    """
    register_name: NotRequired[str | None]
    """
    Specifies if the parameter should be registered in datasets
    using a different name than the parameter's ``full_name``.
    """
    on_set_callback: NotRequired[
        Callable[[ParameterBase, ParameterDataTypeVar], None] | None
    ]
    """
    Callback called when the parameter value is set.
    """
    # Members from ParameterKWArgs are redeclared here
    # so that Sphinx can discover and document them.
    label: NotRequired[str | None]
    """
    Normally used as the axis label when this parameter is graphed,
    along with ``unit``.
    """
    unit: NotRequired[str | None]
    """
    The unit of measure. Use ``''`` for unitless.
    """
    get_cmd: NotRequired[str | Callable[..., Any] | Literal[False] | None]
    """
    A command to issue to the instrument to retrieve the value of this
    parameter. Can be a callable with zero args, a VISA command string,
    ``None`` to use ``get_raw``, or ``False`` to disable getting.
    """
    set_cmd: NotRequired[str | Callable[..., Any] | Literal[False] | None]
    """
    A command to issue to the instrument to set the value of this
    parameter. Can be a callable with one arg, a VISA command string,
    ``None`` to use ``set_raw``, or ``False`` to disable setting.
    Default ``False``.
    """
    initial_value: NotRequired[ParameterDataTypeVar | None]
    """
    Value to set the parameter to at the end of its initialization.
    Cannot be passed together with ``initial_cache_value`` argument.
    """
    docstring: NotRequired[str | None]
    """
    Documentation string for the ``__doc__`` field of the object.
    """
    initial_cache_value: NotRequired[ParameterDataTypeVar | None]
    """
    Value to set the cache of the parameter to at the end of its
    initialization. Cannot be passed together with ``initial_value``
    argument.
    """
    # Members specific to ParameterWithSetpointsKWArgs
    setpoints: NotRequired[Sequence[ParameterBase] | None]
    """
    A list of other parameters that describe the values, names
    and units of the setpoint axes for this parameter.
    """


class ParameterWithSetpoints(
    Parameter[ParameterDataTypeVar, InstrumentTypeVar_co],
    Generic[ParameterDataTypeVar, InstrumentTypeVar_co],
):
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

    Note:
        ``snapshot_get`` and ``snapshot_value`` default to ``False``
        (unlike :class:`Parameter` where they default to ``True``).

    """

    def __init__(
        self,
        name: str,
        *,
        setpoints: Sequence[ParameterBase] | None = None,
        **kwargs: Unpack[ParameterKWArgs[ParameterDataTypeVar, InstrumentTypeVar_co]],
    ) -> None:
        kwargs.setdefault("snapshot_get", False)
        kwargs.setdefault("snapshot_value", False)
        vals = kwargs.get("vals")
        if not isinstance(vals, Arrays):
            raise ValueError(
                f"A ParameterWithSetpoints must have an Arrays "
                f"validator got {type(vals)}"
            )
        if vals.shape_unevaluated is None:
            raise RuntimeError(
                "A ParameterWithSetpoints must have a shape defined for its validator."
            )

        super().__init__(
            name=name,
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

    @property
    def depends_on(self) -> ParameterSet:
        return ParameterSet(self.setpoints)

    def unpack_self(self, value: ValuesType) -> list[tuple[ParameterBase, ValuesType]]:
        unpacked_results: list[tuple[ParameterBase, ValuesType]] = []
        setpoint_params = []
        setpoint_data = []
        for setpointparam in self.setpoints:
            these_setpoints = setpointparam.get()
            setpoint_params.append(setpointparam)
            setpoint_data.append(these_setpoints)
        output_grids = np.meshgrid(*setpoint_data, indexing="ij")
        for param, grid in zip(setpoint_params, output_grids):
            unpacked_results.append((param, grid))
        unpacked_results.extend(
            super().unpack_self(value)
        )  # Must come last to preserve original ordering
        return unpacked_results

    def _set_paramtype(self, paramtype: str) -> None:
        super()._set_paramtype(paramtype)
        for setpoint in self.setpoints:
            setpoint.paramtype = paramtype


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
    if results is not None:
        return parameter.unpack_self(results)
    else:
        return parameter.unpack_self(parameter.get())
