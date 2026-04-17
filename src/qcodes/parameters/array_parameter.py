from __future__ import annotations

import collections.abc
import os
from typing import TYPE_CHECKING, Any, Generic

import numpy as np

try:
    from qcodes_loop.data.data_array import DataArray

    has_loop = True
except ImportError:
    has_loop = False

from typing import NotRequired, TypedDict

from .parameter_base import (
    InstrumentTypeVar_co,
    ParameterBase,
    ParameterBaseKWArgs,
    ParameterDataTypeVar,
)
from .sequence_helpers import is_sequence_of

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from typing_extensions import Unpack

    from qcodes.validators import Validator


try:
    from qcodes_loop.data.data_array import DataArray

    _SP_TYPES: tuple[type, ...] = (
        type(None),
        DataArray,
        collections.abc.Sequence,
        collections.abc.Iterator,
        np.ndarray,
    )
except ImportError:
    _SP_TYPES = (
        type(None),
        collections.abc.Sequence,
        collections.abc.Iterator,
        np.ndarray,
    )


class ArrayParameterKWArgs(
    TypedDict,
    Generic[ParameterDataTypeVar, InstrumentTypeVar_co],
):
    """
    This TypedDict defines the type of the kwargs that can be passed to
    the ``ArrayParameter`` class.

    A subclass of ``ArrayParameter`` should take
    ``**kwargs: Unpack[ArrayParameterKWArgs]`` as input and forward this to
    the super class to ensure that it can accept all the arguments
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
    # Members specific to ArrayParameterKWArgs
    shape: NotRequired[Sequence[int]]
    """
    The shape (as used in numpy arrays) of the array to expect.
    Scalars should be denoted by (), 1D arrays as (n,),
    2D arrays as (n, m), etc.
    """
    label: NotRequired[str | None]
    """
    Normally used as the axis label when this parameter is graphed,
    along with ``unit``.
    """
    unit: NotRequired[str | None]
    """
    The unit of measure. Use ``''`` for unitless.
    """
    setpoints: NotRequired[Sequence[Any] | None]
    """
    The setpoints for each dimension of the returned array.
    """
    setpoint_names: NotRequired[Sequence[str] | None]
    """
    One identifier (like ``name``) per setpoint array.
    """
    setpoint_labels: NotRequired[Sequence[str] | None]
    """
    One label (like ``labels``) per setpoint array.
    """
    setpoint_units: NotRequired[Sequence[str] | None]
    """
    One unit (like ``v``) per setpoint array.
    """
    docstring: NotRequired[str | None]
    """
    Documentation string for the ``__doc__`` field of the object.
    """


class ArrayParameter(
    ParameterBase[ParameterDataTypeVar, InstrumentTypeVar_co],
    Generic[ParameterDataTypeVar, InstrumentTypeVar_co],
):
    """
    A gettable parameter that returns an array of values.
    Not necessarily part of an instrument.

    For new driver we strongly recommend using
    :class:`.ParameterWithSetpoints` which is both more flexible and
    significantly easier to use

    Subclasses should define a ``.get_raw`` method, which returns an array.
    This method is automatically wrapped to provide a ``.get`` method.

    :class:`.ArrayParameter` can be used in both a
    :class:`qcodes.dataset.Measurement`
    as well as in the legacy :class:`qcodes.loops.Loop`
    and :class:`qcodes.measure.Measure` measurements

    When used in a ``Loop`` or ``Measure`` operation, this will be entered
    into a single ``DataArray``, with extra dimensions added by the ``Loop``.
    The constructor args describe the array we expect from each ``.get`` call
    and how it should be handled.

    For now you must specify upfront the array shape, and this cannot change
    from one call to the next. Later we intend to require only that you specify
    the dimension, and the size of each dimension can vary from call to call.

    Args:
        name: The local name of the parameter. Should be a valid
            identifier, i.e. no spaces or special characters. If this parameter
            is part of an ``Instrument`` or ``Station``, this is how it will be
            referenced from that parent, i.e. ``instrument.name`` or
            ``instrument.parameters[name]``

        shape: The shape (as used in numpy arrays) of the array
            to expect. Scalars should be denoted by (), 1D arrays as (n,),
            2D arrays as (n, m), etc.

        label: Normally used as the axis label when this
            parameter is graphed, along with ``unit``.

        unit: The unit of measure. Use ``''`` for unitless.

        setpoints: ``array`` can be a DataArray, numpy.ndarray, or sequence.
            The setpoints for each dimension of the returned array. An
            N-dimension item should have N setpoint arrays, where the first is
            1D, the second 2D, etc.
            If omitted for any or all items, defaults to integers from zero in
            each respective direction.
            **Note**: if the setpoints will be different each measurement,
            leave this out and return the setpoints (with extra names) in
            ``.get``.

        setpoint_names: One identifier (like ``name``) per setpoint array.
            Ignored if a setpoint is a DataArray, which already has a name.

        setpoint_labels: One label (like ``labels``) per setpoint array.
            Ignored if a setpoint is a DataArray, which already has a label.

        setpoint_units: One unit (like ``v``) per setpoint array. Ignored
            if a setpoint is a DataArray, which already has a unit.

        docstring: documentation string for the ``__doc__``
            field of the object. The ``__doc__`` field of the instance
            is used by some help systems, but not all.

        **kwargs: Forwarded to the ``ParameterBase`` base class.
            Note that ``snapshot_value`` defaults to ``False`` for
            ``ArrayParameter``. See :class:`ParameterBaseKWArgs` for
            details.

    """

    def __init__(
        self,
        name: str,
        *,
        shape: Sequence[int],
        label: str | None = None,
        unit: str | None = None,
        setpoints: Sequence[Any] | None = None,
        setpoint_names: Sequence[str] | None = None,
        setpoint_labels: Sequence[str] | None = None,
        setpoint_units: Sequence[str] | None = None,
        docstring: str | None = None,
        **kwargs: Unpack[
            ParameterBaseKWArgs[ParameterDataTypeVar, InstrumentTypeVar_co]
        ],
    ) -> None:
        kwargs.setdefault("snapshot_value", False)
        super().__init__(
            name,
            **kwargs,
        )

        if self.settable:
            # TODO (alexcjohnson): can we support, ala Combine?
            raise AttributeError("ArrayParameters do not support set at this time.")

        self._meta_attrs.extend(
            ["setpoint_names", "setpoint_labels", "setpoint_units", "label", "unit"]
        )

        self.label = name if label is None else label
        self.unit = unit if unit is not None else ""

        nt: type[None] = type(None)

        if not is_sequence_of(shape, int):
            raise ValueError("shapes must be a tuple of ints, not " + repr(shape))
        self.shape = shape

        # require one setpoint per dimension of shape
        sp_shape = (len(shape),)

        if setpoints is not None and not is_sequence_of(
            setpoints, _SP_TYPES, shape=sp_shape
        ):
            raise ValueError("setpoints must be a tuple of arrays")
        if setpoint_names is not None and not is_sequence_of(
            setpoint_names, (nt, str), shape=sp_shape
        ):
            raise ValueError("setpoint_names must be a tuple of strings")
        if setpoint_labels is not None and not is_sequence_of(
            setpoint_labels, (nt, str), shape=sp_shape
        ):
            raise ValueError("setpoint_labels must be a tuple of strings")
        if setpoint_units is not None and not is_sequence_of(
            setpoint_units, (nt, str), shape=sp_shape
        ):
            raise ValueError("setpoint_units must be a tuple of strings")

        self.setpoints = setpoints
        self.setpoint_names = setpoint_names
        self.setpoint_labels = setpoint_labels
        self.setpoint_units = setpoint_units

        self.__doc__ = os.linesep.join(
            (
                "Parameter class:",
                "",
                f"* `name` {self.name}",
                f"* `label` {self.label}",
                f"* `unit` {self.unit}",
                f"* `shape` {self.shape!r}",
            )
        )

        if docstring is not None:
            self.__doc__ = os.linesep.join((docstring, "", self.__doc__))

        if not self.gettable and not self.settable:
            raise AttributeError("ArrayParameter must have a get, set or both")

    @property
    def setpoint_full_names(self) -> Sequence[str] | None:
        """
        Full names of setpoints including instrument names if available
        """
        if self.setpoint_names is None:
            return None
        # omit the last part of name_parts which is the parameter name
        # and not part of the setpoint names
        inst_name = "_".join(self.name_parts[:-1])
        if inst_name != "":
            spnames = []
            for spname in self.setpoint_names:
                if spname is not None:
                    spnames.append(inst_name + "_" + spname)
                else:
                    spnames.append(None)
            return tuple(spnames)
        else:
            return self.setpoint_names
