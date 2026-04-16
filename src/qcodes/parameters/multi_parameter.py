from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, Generic, NotRequired, TypedDict

import numpy as np

from .parameter_base import (
    InstrumentTypeVar_co,
    ParameterBase,
    ParameterBaseKWArgs,
    ParameterDataTypeVar,
)
from .sequence_helpers import is_sequence_of

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from typing_extensions import Unpack

    from .parameter_base import Validator

try:
    from qcodes_loop.data.data_array import DataArray

    _SP_TYPES: tuple[type, ...] = (
        type(None),
        DataArray,
        Sequence,
        Iterator,
        np.ndarray,
    )
except ImportError:
    _SP_TYPES = (
        type(None),
        Sequence,
        Iterator,
        np.ndarray,
    )


def _is_nested_sequence_or_none(
    obj: Any,
    types: type[object | None] | tuple[type[object | None], ...] | None,
    shapes: Sequence[Sequence[int | None]],
) -> bool:
    """Validator for MultiParameter setpoints/names/labels"""
    if obj is None:
        return True

    if not is_sequence_of(obj, tuple, shape=(len(shapes),)):
        return False

    for obji, shapei in zip(obj, shapes):
        if not is_sequence_of(obji, types, shape=(len(shapei),)):
            return False

    return True


class MultiParameterKWArgs(
    TypedDict,
    Generic[ParameterDataTypeVar, InstrumentTypeVar_co],
):
    """
    This TypedDict defines the type of the kwargs that can be passed to
    the ``MultiParameter`` class.

    A subclass of ``MultiParameter`` should take
    ``**kwargs: Unpack[MultiParameterKWArgs]`` as input and forward this to
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
    # Members specific to MultiParameterKWArgs
    names: NotRequired[Sequence[str]]
    """
    A name for each item returned by a ``.get`` call.
    """
    shapes: NotRequired[Sequence[Sequence[int]]]
    """
    The shape (as used in numpy arrays) of each item.
    """
    labels: NotRequired[Sequence[str] | None]
    """
    A label for each item. Normally used as the axis label
    when a component is graphed, along with the matching entry
    from ``units``.
    """
    units: NotRequired[Sequence[str] | None]
    """
    A unit of measure for each item. Use ``''`` or ``None``
    for unitless values.
    """
    setpoints: NotRequired[Sequence[Sequence[Any]] | None]
    """
    The setpoints for each returned array.
    """
    setpoint_names: NotRequired[Sequence[Sequence[str]] | None]
    """
    One identifier (like ``name``) per setpoint array.
    """
    setpoint_labels: NotRequired[Sequence[Sequence[str]] | None]
    """
    One label (like ``labels``) per setpoint array.
    """
    setpoint_units: NotRequired[Sequence[Sequence[str]] | None]
    """
    One unit (like ``V``) per setpoint array.
    """
    docstring: NotRequired[str | None]
    """
    Documentation string for the ``__doc__`` field of the object.
    """


class MultiParameter(
    ParameterBase[ParameterDataTypeVar, InstrumentTypeVar_co],
    Generic[ParameterDataTypeVar, InstrumentTypeVar_co],
):
    """
    A gettable parameter that returns multiple values with separate names,
    each of arbitrary shape. Not necessarily part of an instrument.

    Subclasses should define a ``.get_raw`` method, which returns a sequence of
    values. This method is automatically wrapped to provide a ``.get`` method.
    When used in a legacy  method``Loop`` or ``Measure`` operation, each of
    these values will be entered into a different ``DataArray``. The
    constructor args describe what data we expect from each ``.get`` call
    and how it should be handled. ``.get`` should always return the same
    number of items, and most of the constructor arguments should be tuples
    of that same length.

    For now you must specify upfront the array shape of each item returned by
    ``.get_raw``, and this cannot change from one call to the next. Later, we
    intend to require only that you specify the dimension of each item
    returned, and the size of each dimension can vary from call to call.

    Args:
        name: The local name of the whole parameter. Should be a valid
            identifier, ie no spaces or special characters. If this parameter
            is part of an Instrument or Station, this is how it will be
            referenced from that parent, i.e. ``instrument.name`` or
            ``instrument.parameters[name]``.

        names: A name for each item returned by a ``.get``
            call. Will be used as the basis of the ``DataArray`` names
            when this parameter is used to create a ``DataSet``.

        shapes: The shape (as used in numpy arrays) of
            each item. Scalars should be denoted by (), 1D arrays as (n,),
            2D arrays as (n, m), etc.

        labels: A label for each item. Normally used
            as the axis label when a component is graphed, along with the
            matching entry from ``units``.

        units: A unit of measure for each item.
            Use ``''`` or ``None`` for unitless values.

        setpoints: ``array`` can be a DataArray, numpy.ndarray, or sequence.
            The setpoints for each returned array. An N-dimension item should
            have N setpoint arrays, where the first is 1D, the second 2D, etc.
            If omitted for any or all items, defaults to integers from zero in
            each respective direction.
            **Note**: if the setpoints will be different each measurement,
            leave this out and return the setpoints (with extra names) in
            ``.get``.

        setpoint_names: One identifier (like
            ``name``) per setpoint array. Ignored if a setpoint is a
            DataArray, which already has a name.

        setpoint_labels: One label (like
            ``labels``) per setpoint array. Ignored if a setpoint is a
            DataArray, which already has a label.

        setpoint_units: One unit (like
            ``V``) per setpoint array. Ignored if a setpoint is a
            DataArray, which already has a unit.

        docstring: Documentation string for the ``__doc__``
            field of the object. The ``__doc__`` field of the  instance is
            used by some help systems, but not all

        **kwargs: Forwarded to the ``ParameterBase`` base class.
            Note that ``snapshot_value`` defaults to ``False`` for
            ``MultiParameter``. See :class:`ParameterBaseKWArgs` for
            details.

    """

    def __init__(
        self,
        name: str,
        *,
        names: Sequence[str],
        shapes: Sequence[Sequence[int]],
        labels: Sequence[str] | None = None,
        units: Sequence[str] | None = None,
        setpoints: Sequence[Sequence[Any]] | None = None,
        setpoint_names: Sequence[Sequence[str]] | None = None,
        setpoint_labels: Sequence[Sequence[str]] | None = None,
        setpoint_units: Sequence[Sequence[str]] | None = None,
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

        self._meta_attrs.extend(
            [
                "setpoint_names",
                "setpoint_labels",
                "setpoint_units",
                "names",
                "labels",
                "units",
            ]
        )

        if not is_sequence_of(names, str):
            raise ValueError("names must be a tuple of strings, not " + repr(names))

        self.names = tuple(names)
        self.labels = labels if labels is not None else names
        self.units = units if units is not None else [""] * len(names)

        nt: type[None] = type(None)

        if not is_sequence_of(shapes, int, depth=2) or len(shapes) != len(names):
            raise ValueError(
                f"shapes must be a tuple of tuples of ints, not {shapes!r}"
            )
        self.shapes = shapes

        if not _is_nested_sequence_or_none(setpoints, _SP_TYPES, shapes):
            raise ValueError("setpoints must be a tuple of tuples of arrays")

        if not _is_nested_sequence_or_none(setpoint_names, (nt, str), shapes):
            raise ValueError("setpoint_names must be a tuple of tuples of strings")

        if not _is_nested_sequence_or_none(setpoint_labels, (nt, str), shapes):
            raise ValueError("setpoint_labels must be a tuple of tuples of strings")

        if not _is_nested_sequence_or_none(setpoint_units, (nt, str), shapes):
            raise ValueError("setpoint_units must be a tuple of tuples of strings")

        self.setpoints = setpoints
        self.setpoint_names = setpoint_names
        self.setpoint_labels = setpoint_labels
        self.setpoint_units = setpoint_units

        self.__doc__ = os.linesep.join(
            (
                "MultiParameter class:",
                "",
                f"* `name` {self.name}",
                "* `names` {}".format(", ".join(self.names)),
                "* `labels` {}".format(", ".join(self.labels)),
                "* `units` {}".format(", ".join(self.units)),
            )
        )

        if docstring is not None:
            self.__doc__ = os.linesep.join((docstring, "", self.__doc__))

        if not self.gettable and not self.settable:
            raise AttributeError("MultiParameter must have a get, set or both")

    @property
    def short_names(self) -> tuple[str, ...]:
        """
        short_names is identical to names i.e. the names of the parameter
        parts but does not add the instrument name.

        It exists for consistency with instruments and other parameters.
        """

        return self.names

    @property
    def full_names(self) -> tuple[str, ...]:
        """
        Names of the parameter components including the name of the instrument
        and submodule that the parameter may be bound to. The name parts are
        separated by underscores, like this: ``instrument_submodule_parameter``
        """
        inst_name = "_".join(self.name_parts[:-1])
        if inst_name != "":
            return tuple(inst_name + "_" + name for name in self.names)
        else:
            return self.names

    @property
    def setpoint_full_names(self) -> Sequence[Sequence[str]] | None:
        """
        Full names of setpoints including instrument names, if available
        """
        if self.setpoint_names is None:
            return None
        # omit the last part of name_parts which is the parameter name
        # and not part of the setpoint names
        inst_name = "_".join(self.name_parts[:-1])
        if inst_name != "":
            full_sp_names = []
            for sp_group in self.setpoint_names:
                full_sp_names_subgroupd = []
                for spname in sp_group:
                    if spname is not None:
                        full_sp_names_subgroupd.append(inst_name + "_" + spname)
                    else:
                        full_sp_names_subgroupd.append(None)
                full_sp_names.append(tuple(full_sp_names_subgroupd))

            return tuple(full_sp_names)
        else:
            return self.setpoint_names
