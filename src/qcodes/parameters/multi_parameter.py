from __future__ import annotations

import os
import warnings
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, ClassVar, Generic

import numpy as np

from qcodes.utils import QCoDeSDeprecationWarning

from .parameter_base import InstrumentTypeVar_co, ParameterBase, ParameterDataTypeVar
from .sequence_helpers import is_sequence_of

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

        instrument: The instrument this parameter
            belongs to, if any.

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

        snapshot_get: Prevent any update to the parameter, for example
            if it takes too long to update. Default ``True``.

        snapshot_value: Should the value of the parameter be stored in the
            snapshot. Unlike Parameter this defaults to False as
            MultiParameters are potentially huge.

        snapshot_exclude: True prevents parameter to be
            included in the snapshot. Useful if there are many of the same
            parameter which are clogging up the snapshot.
            Default ``False``.

        metadata: Extra information to include with the
            JSON snapshot of the parameter.

    """

    _NAMES_UNSET: Any = object()
    _SHAPES_UNSET: Any = object()

    _DEPRECATED_POSITIONAL_ARGS: ClassVar[tuple[str, ...]] = (
        "names",
        "shapes",
        "instrument",
        "labels",
        "units",
        "setpoints",
        "setpoint_names",
        "setpoint_labels",
        "setpoint_units",
        "docstring",
        "snapshot_get",
        "snapshot_value",
        "snapshot_exclude",
        "metadata",
    )

    def __init__(
        self,
        name: str,
        *args: Any,
        names: Sequence[str] = _NAMES_UNSET,  # type: ignore[assignment]
        shapes: Sequence[Sequence[int]] = _SHAPES_UNSET,  # type: ignore[assignment]
        # mypy seems to be confused here. The bound and default for InstrumentTypeVar_co
        # contains None but mypy will not allow it as a default as of v 1.19.0
        instrument: InstrumentTypeVar_co = None,  # type: ignore[assignment]
        labels: Sequence[str] | None = None,
        units: Sequence[str] | None = None,
        setpoints: Sequence[Sequence[Any]] | None = None,
        setpoint_names: Sequence[Sequence[str]] | None = None,
        setpoint_labels: Sequence[Sequence[str]] | None = None,
        setpoint_units: Sequence[Sequence[str]] | None = None,
        docstring: str | None = None,
        snapshot_get: bool = True,
        snapshot_value: bool = False,
        snapshot_exclude: bool = False,
        metadata: Mapping[Any, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if args:
            # TODO: After QCoDeS 0.57 remove the args argument and delete this code block.
            positional_names = self._DEPRECATED_POSITIONAL_ARGS
            if len(args) > len(positional_names):
                raise TypeError(
                    f"{type(self).__name__}.__init__() takes at most "
                    f"{len(positional_names) + 2} positional arguments "
                    f"({len(args) + 2} given)"
                )

            _defaults: dict[str, Any] = {
                "names": self._NAMES_UNSET,
                "shapes": self._SHAPES_UNSET,
                "instrument": None,
                "labels": None,
                "units": None,
                "setpoints": None,
                "setpoint_names": None,
                "setpoint_labels": None,
                "setpoint_units": None,
                "docstring": None,
                "snapshot_get": True,
                "snapshot_value": False,
                "snapshot_exclude": False,
                "metadata": None,
            }

            _kwarg_vals: dict[str, Any] = {
                "names": names,
                "shapes": shapes,
                "instrument": instrument,
                "labels": labels,
                "units": units,
                "setpoints": setpoints,
                "setpoint_names": setpoint_names,
                "setpoint_labels": setpoint_labels,
                "setpoint_units": setpoint_units,
                "docstring": docstring,
                "snapshot_get": snapshot_get,
                "snapshot_value": snapshot_value,
                "snapshot_exclude": snapshot_exclude,
                "metadata": metadata,
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
            names = _pos.get("names", names)
            shapes = _pos.get("shapes", shapes)
            instrument = _pos.get("instrument", instrument)
            labels = _pos.get("labels", labels)
            units = _pos.get("units", units)
            setpoints = _pos.get("setpoints", setpoints)
            setpoint_names = _pos.get("setpoint_names", setpoint_names)
            setpoint_labels = _pos.get("setpoint_labels", setpoint_labels)
            setpoint_units = _pos.get("setpoint_units", setpoint_units)
            docstring = _pos.get("docstring", docstring)
            snapshot_get = _pos.get("snapshot_get", snapshot_get)
            snapshot_value = _pos.get("snapshot_value", snapshot_value)
            snapshot_exclude = _pos.get("snapshot_exclude", snapshot_exclude)
            metadata = _pos.get("metadata", metadata)

        if names is self._NAMES_UNSET:
            raise TypeError(
                f"{type(self).__name__}.__init__() missing required "
                f"keyword argument: 'names'"
            )
        if shapes is self._SHAPES_UNSET:
            raise TypeError(
                f"{type(self).__name__}.__init__() missing required "
                f"keyword argument: 'shapes'"
            )

        super().__init__(
            name,
            instrument=instrument,
            snapshot_get=snapshot_get,
            metadata=metadata,
            snapshot_value=snapshot_value,
            snapshot_exclude=snapshot_exclude,
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
