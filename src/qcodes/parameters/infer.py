from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from qcodes.instrument import Instrument, InstrumentBase, InstrumentChannel
from qcodes.instrument.parameter import DelegateParameter, Parameter

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


class InferError(AttributeError): ...


class InferAttrs:
    """Holds a global set of attribute name that will be inferred"""

    _known_attrs: ClassVar[set[str]] = set()

    @classmethod
    def add_attr(cls, attr: str) -> None:
        cls._known_attrs.add(attr)

    @classmethod
    def known_attrs(cls) -> tuple[str, ...]:
        return tuple(cls._known_attrs)

    @classmethod
    def discard_attr(cls, attr: str) -> None:
        cls._known_attrs.discard(attr)

    @classmethod
    def clear_attrs(cls) -> None:
        cls._known_attrs = set()


def get_root_param(
    param: Parameter | DelegateParameter | None,
    parent_param: Parameter | None = None,
    alt_source_attrs: Sequence[str] | None = None,
) -> Parameter:
    """Return the root parameter in a chain of DelegateParameters or other linking Parameters

    This method recursively searches on the initial parameter.
    - If the parameter is a DelegateParameter, it returns the .source.
    - If the parameter is not a DelegateParameter, but has an attribute in
    either alt_source_attrs or the InferAttrs class which is a parameter,
    then it returns that parameter
    - If the parameter is None, because the previous DelegateParameter did not have a source
    it raises an InferError


    """
    parent_param = param if parent_param is None else parent_param
    if alt_source_attrs is None:
        alt_source_attrs_set: Iterable[str] = InferAttrs.known_attrs()
    else:
        alt_source_attrs_set = set.union(
            set(alt_source_attrs), set(InferAttrs.known_attrs())
        )

    if param is None:
        raise InferError(f"Parameter {parent_param} is not attached to a source")
    if isinstance(param, DelegateParameter):
        return get_root_param(param.source, parent_param)
    for alt_source_attr in alt_source_attrs_set:
        alt_source = getattr(param, alt_source_attr, None)
        if alt_source is not None and isinstance(alt_source, Parameter):
            return get_root_param(
                alt_source, parent_param=parent_param, alt_source_attrs=alt_source_attrs
            )
    return param


def infer_instrument(
    param: Parameter,
    alt_source_attrs: Sequence[str] | None = None,
) -> InstrumentBase:
    """Find the instrument that owns a parameter or delegate parameter."""
    root_param = get_root_param(param, alt_source_attrs=alt_source_attrs)
    instrument = get_instrument_from_param(root_param)
    if isinstance(instrument, InstrumentChannel):
        return instrument.root_instrument
    elif isinstance(instrument, Instrument):
        return instrument

    raise InferError(f"Could not determine source instrument for parameter {param}")


def infer_channel(
    param: Parameter,
    alt_source_attrs: Sequence[str] | None = None,
) -> InstrumentChannel:
    """Find the instrument module that owns a parameter."""
    root_param = get_root_param(param, alt_source_attrs=alt_source_attrs)
    channel = get_instrument_from_param(root_param)
    if isinstance(channel, InstrumentChannel):
        return channel
    raise InferError(
        f"Could not determine a root instrument channel for parameter {param}"
    )


def get_instrument_from_param(
    param: Parameter,
) -> InstrumentBase:
    if param.instrument is not None:
        return param.instrument
    raise InferError(f"Parameter {param} has no instrument")
