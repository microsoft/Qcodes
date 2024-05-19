from __future__ import annotations

import collections.abc
import os
from typing import TYPE_CHECKING, Any

import numpy as np

try:
    from qcodes_loop.data.data_array import DataArray

    has_loop = True
except ImportError:
    has_loop = False

from .parameter_base import ParameterBase
from .sequence_helpers import is_sequence_of

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from qcodes.instrument import InstrumentBase


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


class ArrayParameter(ParameterBase):
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

    def __init__(
        self,
        name: str,
        shape: Sequence[int],
        instrument: InstrumentBase | None = None,
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
