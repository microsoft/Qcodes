from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

from qcodes.instrument import Instrument, InstrumentBase, InstrumentModule
from qcodes.instrument.parameter import DelegateParameter, Parameter

if TYPE_CHECKING:
    from collections.abc import Iterable

DOES_NOT_EXIST = "Does not exist"


class InferError(AttributeError): ...


class InferAttrs:
    """Holds a global set of attribute name that will be inferred"""

    _known_attrs: ClassVar[set[str]] = set()

    @classmethod
    def add_attrs(cls, attrs: str | Iterable[str]) -> None:
        if isinstance(attrs, str):
            attrs = (attrs,)
        cls._known_attrs.update(set(attrs))

    @classmethod
    def known_attrs(cls) -> tuple[str, ...]:
        return tuple(cls._known_attrs)

    @classmethod
    def discard_attr(cls, attr: str) -> None:
        cls._known_attrs.discard(attr)

    @classmethod
    def clear_attrs(cls) -> None:
        cls._known_attrs = set()


def get_root_parameter(
    param: Parameter,
    alt_source_attrs: Sequence[str] | None = None,
) -> Parameter:
    """Return the root parameter in a chain of DelegateParameters or other linking Parameters"""
    alt_source_attrs_set = _merge_user_and_class_attrs(alt_source_attrs)

    if isinstance(param, DelegateParameter):
        if param.source is None:
            raise InferError(f"Parameter {param} is not attached to a source")
        return get_root_parameter(param.source)

    for alt_source_attr in alt_source_attrs_set:
        alt_source = getattr(param, alt_source_attr, DOES_NOT_EXIST)
        if alt_source is None:
            raise InferError(
                f"Parameter {param} is not attached to a source on attribute {alt_source_attr}"
            )
        elif isinstance(alt_source, Parameter):
            return get_root_parameter(alt_source, alt_source_attrs=alt_source_attrs)
    return param


def infer_instrument(
    param: Parameter,
    alt_source_attrs: Sequence[str] | None = None,
) -> InstrumentBase:
    """Find the instrument that owns a parameter or delegate parameter."""
    root_param = get_root_parameter(param, alt_source_attrs=alt_source_attrs)
    instrument = get_instrument_from_param(root_param)
    if isinstance(instrument, InstrumentModule):
        return instrument.root_instrument
    elif isinstance(instrument, Instrument):
        return instrument

    raise InferError(f"Could not determine source instrument for parameter {param}")


def infer_channel(
    param: Parameter,
    alt_source_attrs: Sequence[str] | None = None,
) -> InstrumentModule:
    """Find the instrument module that owns a parameter or delegate parameter"""
    root_param = get_root_parameter(param, alt_source_attrs=alt_source_attrs)
    channel = get_instrument_from_param(root_param)
    if isinstance(channel, InstrumentModule):
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


def get_parameter_chain(
    param_chain: Parameter | Sequence[Parameter],
    alt_source_attrs: str | Sequence[str] | None = None,
) -> tuple[Parameter, ...]:
    """Return the chain of DelegateParameters or other linking Parameters"""
    alt_source_attrs_set = _merge_user_and_class_attrs(alt_source_attrs)

    if not isinstance(param_chain, Sequence):
        param_chain = (param_chain,)

    param = param_chain[-1]
    mutable_param_chain = list(param_chain)
    if isinstance(param, DelegateParameter):
        if param.source is None:
            return tuple(param_chain)
        mutable_param_chain.append(param.source)
        return get_parameter_chain(
            mutable_param_chain,
            alt_source_attrs=alt_source_attrs,
        )

    for alt_source_attr in alt_source_attrs_set:
        alt_source = getattr(param, alt_source_attr, DOES_NOT_EXIST)
        if alt_source is None:
            return tuple(param_chain)
        elif isinstance(alt_source, Parameter):
            mutable_param_chain.append(alt_source)
            return get_parameter_chain(
                mutable_param_chain,
                alt_source_attrs=alt_source_attrs,
            )
    return tuple(param_chain)


def _merge_user_and_class_attrs(
    alt_source_attrs: str | Sequence[str] | None = None,
) -> Iterable[str]:
    if alt_source_attrs is None:
        return InferAttrs.known_attrs()
    elif isinstance(alt_source_attrs, str):
        return set.union(set((alt_source_attrs,)), set(InferAttrs.known_attrs()))
    else:
        return set.union(set(alt_source_attrs), set(InferAttrs.known_attrs()))
