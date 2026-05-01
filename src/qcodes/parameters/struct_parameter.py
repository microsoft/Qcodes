"""Parameter that returns structured data as a dataclass or Pydantic BaseModel.

A :class:`StructParameter` wraps a ``get_raw`` that returns a dataclass or
Pydantic v2 :class:`~pydantic.BaseModel` instance. Each field of the struct
is automatically unpacked into a separate dataset column when used in a
:class:`~qcodes.dataset.Measurement`.

Pydantic support is optional and is enabled when ``pydantic`` is installed.
Only Pydantic v2 (``pydantic.BaseModel`` with ``model_fields``) is supported.
"""

from __future__ import annotations

import dataclasses
import os
import typing
from typing import TYPE_CHECKING, Any, Generic

import numpy as np

from qcodes.validators import Arrays, ComplexNumbers, Numbers, Strings

from .parameter_base import (
    InstrumentTypeVar_co,
    ParameterBase,
    ParameterBaseKWArgs,
    ParameterDataTypeVar,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from typing_extensions import Unpack

    from qcodes.dataset.data_set_protocol import ValuesType
    from qcodes.validators import Validator


# Supported paramtypes for dataset columns
_ALLOWED_PARAMTYPES = frozenset({"numeric", "text", "complex", "array"})


def _is_pydantic_model_class(cls: type) -> bool:
    """Check if a class is a Pydantic v2 BaseModel subclass."""
    try:
        from pydantic import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
            BaseModel,
        )

        return isinstance(cls, type) and issubclass(cls, BaseModel)
    except ImportError:
        return False


def _get_struct_fields(struct_type: type) -> list[tuple[str, type]]:
    """Extract (name, annotation) pairs from a dataclass or Pydantic model.

    Args:
        struct_type: A dataclass or Pydantic v2 BaseModel class.

    Returns:
        List of ``(field_name, field_type)`` tuples.

    Raises:
        TypeError: If ``struct_type`` is neither a dataclass nor a Pydantic
            BaseModel.

    """
    if dataclasses.is_dataclass(struct_type):
        # Use get_type_hints to resolve string annotations from
        # `from __future__ import annotations`
        try:
            hints = typing.get_type_hints(struct_type)
        except Exception:
            hints = {}
        return [
            (f.name, hints.get(f.name, object))
            for f in dataclasses.fields(struct_type)  # type: ignore[arg-type]
        ]

    if _is_pydantic_model_class(struct_type):
        # Pydantic v2 resolves annotations itself, so model_fields
        # already has the resolved type in field.annotation
        fields = struct_type.model_fields  # type: ignore[union-attr]
        result: list[tuple[str, type]] = []
        for name, field in fields.items():
            ann: type = (
                field.annotation if isinstance(field.annotation, type) else object
            )
            result.append((name, ann))
        return result

    raise TypeError(
        f"struct_type must be a dataclass or Pydantic v2 BaseModel, got {struct_type!r}"
    )


def _infer_paramtype_from_annotation(annotation: type) -> str:
    """Map a Python type annotation to a QCoDeS paramtype string.

    Args:
        annotation: The type annotation of a struct field.

    Returns:
        One of ``"numeric"``, ``"text"``, ``"complex"``, or ``"array"``.

    Raises:
        TypeError: If the annotation maps to an unsupported or nested type.

    """
    # Handle basic types
    if annotation in (float, int, bool):
        return "numeric"
    if annotation is str:
        return "text"
    if annotation is complex:
        return "complex"
    if annotation is np.ndarray:
        return "array"

    # Reject nested dataclasses and Pydantic models
    if dataclasses.is_dataclass(annotation) or _is_pydantic_model_class(annotation):
        raise TypeError(
            f"Nested structured types are not supported as struct fields: "
            f"{annotation!r}"
        )

    # Default to numeric for unknown types (int subclasses, enums, etc.)
    return "numeric"


def _validator_for_paramtype(paramtype: str) -> Validator[Any]:
    """Create a QCoDeS validator matching the given paramtype."""
    match paramtype:
        case "numeric":
            return Numbers()
        case "text":
            return Strings()
        case "complex":
            return ComplexNumbers()
        case "array":
            return Arrays()
        case _:
            raise ValueError(f"Unknown paramtype: {paramtype!r}")


def _extract_field_value(struct_instance: Any, field_name: str) -> Any:
    """Extract a field value from a dataclass or Pydantic model instance."""
    return getattr(struct_instance, field_name)


class _FieldParameter(ParameterBase[Any, None]):
    """Synthetic parameter representing a single field of a StructParameter.

    These parameters are not independently gettable or settable. They exist
    solely for dataset registration and field-value storage during unpacking.
    """

    def __init__(
        self,
        name: str,
        *,
        label: str | None = None,
        unit: str | None = None,
        paramtype: str = "numeric",
    ) -> None:
        super().__init__(
            name,
            bind_to_instrument=False,
            snapshot_value=False,
        )
        self.label = label or name
        self.unit = unit or ""
        self._set_paramtype(paramtype)


class StructParameter(
    ParameterBase[ParameterDataTypeVar, InstrumentTypeVar_co],
    Generic[ParameterDataTypeVar, InstrumentTypeVar_co],
):
    """A gettable parameter that returns a dataclass or Pydantic BaseModel.

    When used in a :class:`~qcodes.dataset.Measurement`, each field of the
    struct is automatically unpacked into a separate dataset column.

    Subclasses should define a :meth:`get_raw` method that returns an instance
    of ``struct_type``.

    Args:
        name: The local name of the parameter. Must be a valid identifier.
        struct_type: A dataclass or Pydantic v2 BaseModel class whose fields
            define the structure of the returned data.
        field_labels: Optional mapping of ``{field_name: label}`` for
            dataset/graph axis labels. Defaults to field names.
        field_units: Optional mapping of ``{field_name: unit}`` for
            dataset/graph axis units. Defaults to empty strings.
        field_paramtypes: Optional mapping of ``{field_name: paramtype}``
            to override the auto-inferred paramtype. Valid values are
            ``"numeric"``, ``"text"``, ``"complex"``, and ``"array"``.
        docstring: Documentation string for the ``__doc__`` field.
        **kwargs: Forwarded to :class:`ParameterBase`.
            See :class:`ParameterBaseKWArgs` for details.

    Example:

        .. code-block:: python

            from dataclasses import dataclass
            from qcodes.parameters import StructParameter

            @dataclass
            class IVResult:
                voltage: float
                current: float

            class MyIVParameter(StructParameter):
                def get_raw(self):
                    v = self.instrument.ask("MEAS:VOLT?")
                    i = self.instrument.ask("MEAS:CURR?")
                    return IVResult(voltage=float(v), current=float(i))

            param = MyIVParameter(
                "iv_measurement",
                struct_type=IVResult,
                field_units={"voltage": "V", "current": "A"},
            )

    """

    def __init__(
        self,
        name: str,
        struct_type: type,
        *,
        field_labels: Mapping[str, str] | None = None,
        field_units: Mapping[str, str] | None = None,
        field_paramtypes: Mapping[str, str] | None = None,
        docstring: str | None = None,
        **kwargs: Unpack[
            ParameterBaseKWArgs[ParameterDataTypeVar, InstrumentTypeVar_co]
        ],
    ) -> None:
        kwargs.setdefault("snapshot_value", False)
        super().__init__(name, **kwargs)

        self._struct_type = struct_type
        field_labels = field_labels or {}
        field_units = field_units or {}
        field_paramtypes = field_paramtypes or {}

        # Introspect the struct type
        fields = _get_struct_fields(struct_type)
        if not fields:
            raise TypeError(f"struct_type {struct_type.__name__} has no fields")

        # Validate user-supplied overrides reference real fields
        field_name_set = {f[0] for f in fields}
        for mapping_name, mapping in [
            ("field_labels", field_labels),
            ("field_units", field_units),
            ("field_paramtypes", field_paramtypes),
        ]:
            unknown = set(mapping.keys()) - field_name_set
            if unknown:
                raise ValueError(
                    f"{mapping_name} contains unknown field names: {unknown}"
                )

        for pt_name, pt_val in field_paramtypes.items():
            if pt_val not in _ALLOWED_PARAMTYPES:
                raise ValueError(
                    f"Invalid paramtype {pt_val!r} for field {pt_name!r}. "
                    f"Allowed values: {sorted(_ALLOWED_PARAMTYPES)}"
                )

        # Build child parameters for each field
        self._field_parameters: dict[str, _FieldParameter] = {}
        names_list: list[str] = []
        labels_list: list[str] = []
        units_list: list[str] = []

        for field_name, field_annotation in fields:
            paramtype = field_paramtypes.get(
                field_name,
                _infer_paramtype_from_annotation(field_annotation),
            )
            label = field_labels.get(field_name, field_name)
            unit = field_units.get(field_name, "")
            child_name = f"{name}_{field_name}"

            child_param: _FieldParameter = _FieldParameter(
                name=child_name,
                label=label,
                unit=unit,
                paramtype=paramtype,
            )
            child_param.vals = _validator_for_paramtype(paramtype)
            self._field_parameters[field_name] = child_param
            names_list.append(field_name)
            labels_list.append(label)
            units_list.append(unit)

        self.names: tuple[str, ...] = tuple(names_list)
        self.labels: tuple[str, ...] = tuple(labels_list)
        self.units: tuple[str, ...] = tuple(units_list)

        self._meta_attrs.extend(["names", "labels", "units", "struct_type_name"])

        # Generate docstring
        self.__doc__ = os.linesep.join(
            (
                "StructParameter class:",
                "",
                f"* `name` {self.name}",
                f"* `struct_type` {struct_type.__name__}",
                "* `names` {}".format(", ".join(self.names)),
                "* `labels` {}".format(", ".join(self.labels)),
                "* `units` {}".format(", ".join(self.units)),
            )
        )
        if docstring is not None:
            self.__doc__ = os.linesep.join((docstring, "", self.__doc__))

        if not self.gettable:
            raise AttributeError("StructParameter must have a get method")

    @property
    def struct_type(self) -> type:
        """The dataclass or Pydantic BaseModel class for this parameter."""
        return self._struct_type

    @property
    def struct_type_name(self) -> str:
        """Name of the struct type, included in snapshots."""
        return self._struct_type.__name__

    @property
    def field_parameters(self) -> dict[str, _FieldParameter]:
        """Mapping of field name to the synthetic child parameter."""
        return dict(self._field_parameters)

    @property
    def short_names(self) -> tuple[str, ...]:
        """Short names of the struct fields (without instrument prefix)."""
        return self.names

    @property
    def full_names(self) -> tuple[str, ...]:
        """Full names of fields including instrument name prefix."""
        inst_name = "_".join(self.name_parts[:-1])
        if inst_name:
            return tuple(f"{inst_name}_{self.name}_{n}" for n in self.names)
        return tuple(f"{self.name}_{n}" for n in self.names)

    def unpack_self(
        self, value: ValuesType
    ) -> list[tuple[ParameterBase[Any, Any], ValuesType]]:
        """Unpack a struct value into individual field parameter results.

        This method does NOT include the parent struct parameter itself in the
        results. Only the individual field values are returned, each paired
        with its corresponding synthetic child parameter.

        Args:
            value: An instance of the struct type returned by ``get_raw``.

        Returns:
            A list of ``(field_parameter, field_value)`` tuples.

        """
        results: list[tuple[ParameterBase[Any, Any], ValuesType]] = []
        for field_name, field_param in self._field_parameters.items():
            field_value = _extract_field_value(value, field_name)
            results.append((field_param, field_value))
        return results
