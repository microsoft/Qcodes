from __future__ import annotations

import collections.abc
import os
import warnings
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

try:
    from qcodes_loop.data.data_array import DataArray

    has_loop = True
except ImportError:
    has_loop = False
from typing import Generic

from qcodes.utils import QCoDeSDeprecationWarning

from .parameter_base import InstrumentTypeVar_co, ParameterBase, ParameterDataTypeVar
from .sequence_helpers import is_sequence_of

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


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


_SHAPE_UNSET: Any = object()


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

        instrument: The instrument this parameter
            belongs to, if any.

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

        snapshot_get: Prevent any update to the parameter, for example
            if it takes too long to update. Default ``True``.

        snapshot_value: Should the value of the parameter be stored in the
            snapshot. Unlike Parameter this defaults to False as
            ArrayParameters are potentially huge.

        snapshot_exclude: ``True`` prevents parameter to be
            included in the snapshot. Useful if there are many of the same
            parameter which are clogging up the snapshot.

            Default ``False``.

        metadata: Extra information to include with the
            JSON snapshot of the parameter.

    """

    _DEPRECATED_POSITIONAL_ARGS: ClassVar[tuple[str, ...]] = (
        "shape",
        "instrument",
        "label",
        "unit",
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
        shape: Sequence[int] = _SHAPE_UNSET,
        # mypy seems to be confused here. The bound and default for InstrumentTypeVar_co
        # contains None but mypy will not allow it as a default as of v 1.19.0
        instrument: InstrumentTypeVar_co = None,  # type: ignore[assignment]
        label: str | None = None,
        unit: str | None = None,
        setpoints: Sequence[Any] | None = None,
        setpoint_names: Sequence[str] | None = None,
        setpoint_labels: Sequence[str] | None = None,
        setpoint_units: Sequence[str] | None = None,
        docstring: str | None = None,
        snapshot_get: bool = True,
        snapshot_value: bool = False,
        snapshot_exclude: bool = False,
        metadata: Mapping[Any, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if args:
            # TODO: After QCoDeS 0.57 remove the args argument and delete this code block.
            positional_names = __class__._DEPRECATED_POSITIONAL_ARGS
            if len(args) > len(positional_names):
                raise TypeError(
                    f"{type(self).__name__}.__init__() takes at most "
                    f"{len(positional_names) + 2} positional arguments "
                    f"({len(args) + 2} given)"
                )

            _defaults: dict[str, Any] = {
                "shape": _SHAPE_UNSET,
                "instrument": None,
                "label": None,
                "unit": None,
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
                "shape": shape,
                "instrument": instrument,
                "label": label,
                "unit": unit,
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
            shape = _pos.get("shape", shape)
            instrument = _pos.get("instrument", instrument)
            label = _pos.get("label", label)
            unit = _pos.get("unit", unit)
            setpoints = _pos.get("setpoints", setpoints)
            setpoint_names = _pos.get("setpoint_names", setpoint_names)
            setpoint_labels = _pos.get("setpoint_labels", setpoint_labels)
            setpoint_units = _pos.get("setpoint_units", setpoint_units)
            docstring = _pos.get("docstring", docstring)
            snapshot_get = _pos.get("snapshot_get", snapshot_get)
            snapshot_value = _pos.get("snapshot_value", snapshot_value)
            snapshot_exclude = _pos.get("snapshot_exclude", snapshot_exclude)
            metadata = _pos.get("metadata", metadata)

        if shape is _SHAPE_UNSET:
            raise TypeError(
                f"{type(self).__name__}.__init__() missing required "
                f"keyword argument: 'shape'"
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
