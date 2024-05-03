from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

from qcodes.instrument import Instrument, InstrumentBase, InstrumentModule
from qcodes.parameters import DelegateParameter, Parameter

if TYPE_CHECKING:
    from collections.abc import Iterable

DOES_NOT_EXIST = "Does not exist"


class InferError(AttributeError): ...


class InferAttrs:
    """Holds a global set of attribute name that will be inferred"""

    _known_attrs: ClassVar[set[str]] = set()

    @classmethod
    def add(cls, attrs: str | Iterable[str]) -> None:
        if isinstance(attrs, str):
            attrs = (attrs,)
        cls._known_attrs.update(set(attrs))

    @classmethod
    def known_attrs(cls) -> tuple[str, ...]:
        return tuple(cls._known_attrs)

    @classmethod
    def discard(cls, attr: str) -> None:
        cls._known_attrs.discard(attr)

    @classmethod
    def clear(cls) -> None:
        cls._known_attrs = set()


def get_root_parameter(
    param: Parameter,
    alt_source_attrs: Sequence[str] | None = None,
) -> Parameter:
    """
    Return the root parameter in a chain of DelegateParameters or other linking Parameters

    This method calls get_parameter_chain and then checks for various error conditions
    Args:
        param: The DelegateParameter or other linking parameter to find the root parameter from
        alt_source_attrs: The attribute names for custom linking parameters

    Raises:
        InferError: If the linking parameters do not end with a non-linking parameter
        InferError: If the chain of linking parameters loops on itself
    """

    parameter_chain = get_parameter_chain(param, alt_source_attrs)
    root_param = parameter_chain[-1]

    if root_param is parameter_chain[0] and len(parameter_chain) > 1:
        raise InferError(f"{param} generated a loop of linking parameters")
    if isinstance(root_param, DelegateParameter):
        raise InferError(f"Parameter {param} is not attached to a source")

    alt_source_attrs_set = _merge_user_and_class_attrs(alt_source_attrs)
    for alt_source_attr in alt_source_attrs_set:
        alt_source = getattr(param, alt_source_attr, DOES_NOT_EXIST)
        if alt_source is None:
            raise InferError(
                f"Parameter {param} is not attached to a source on attribute {alt_source_attr}"
            )
    return root_param


def infer_instrument(
    param: Parameter,
    alt_source_attrs: Sequence[str] | None = None,
) -> InstrumentBase:
    """
    Find the instrument that owns a parameter or delegate parameter.

    Args:
        param: The DelegateParameter or other linking parameter to find the instrument from
        alt_source_attrs: The attribute names for custom linking parameters

    Raises:
        InferError: If the linking parameters do not end with a non-linking parameter
        InferError: If the instrument of the root parameter is None
        InferError: If the instrument of the root parameter is not an instance of Instrument
    """
    root_param = get_root_parameter(param, alt_source_attrs=alt_source_attrs)
    instrument = get_instrument_from_param(root_param)
    if isinstance(instrument, InstrumentModule):
        return instrument.root_instrument
    elif isinstance(instrument, Instrument):
        return instrument

    raise InferError(f"Could not determine source instrument for parameter {param}")


def infer_instrument_module(
    param: Parameter,
    alt_source_attrs: Sequence[str] | None = None,
) -> InstrumentModule:
    """
    Find the instrument module that owns a parameter or delegate parameter

    Args:
        param: The DelegateParameter or other linking parameter to find the instrument module from
        alt_source_attrs: The attribute names for custom linking parameters

    Raises:
        InferError: If the linking parameters do not end with a non-linking parameter
        InferError: If the instrument module of the root parameter is None
        InferError: If the instrument module of the root parameter is not an instance of InstrumentModule
    """
    root_param = get_root_parameter(param, alt_source_attrs=alt_source_attrs)
    channel = get_instrument_from_param(root_param)
    if isinstance(channel, InstrumentModule):
        return channel
    raise InferError(
        f"Could not determine a root instrument channel for parameter {param}"
    )


def infer_channel(
    param: Parameter,
    alt_source_attrs: Sequence[str] | None = None,
) -> InstrumentModule:
    """An alias for infer_instrument_module"""
    return infer_instrument_module(param, alt_source_attrs)


def get_instrument_from_param(
    param: Parameter,
) -> InstrumentBase:
    """
    Return the instrument attribute from a parameter

    Args:
        param: The parameter to get the instrument module from

    Raises:
        InferError: If the parameter does not have an instrument
    """
    if param.instrument is not None:
        return param.instrument
    raise InferError(f"Parameter {param} has no instrument")


def get_parameter_chain(
    param_chain: Parameter | Sequence[Parameter],
    alt_source_attrs: str | Sequence[str] | None = None,
) -> tuple[Parameter, ...]:
    """
    Return the chain of DelegateParameters or other linking Parameters

    This method traverses singly-linked parameters and returns the resulting chain
    If the parameters loop, then the first and last linking parameters in the chain
    will be identical. Otherwise, the chain starts with the initial argument passed
    and ends when the chain terminates in either a non-linking parameter or a
    linking parameter that links to None

    The search prioritizes the `source` attribute of DelegateParameters first, and
    then looks for other linking attributes in undetermined order.

    Args:
        param_chain: The initial linking parameter or a List linking parameters
            from which to return the chain
        alt_source_attrs: The attribute names for custom linking parameters
    """

    alt_source_attrs_set = _merge_user_and_class_attrs(alt_source_attrs)

    if not isinstance(param_chain, Sequence):
        param_chain = (param_chain,)

    param = param_chain[-1]
    mutable_param_chain = list(param_chain)
    if isinstance(param, DelegateParameter):
        if param.source is None:
            return tuple(param_chain)
        mutable_param_chain.append(param.source)
        if param.source in param_chain:  # There is a loop in the links
            return tuple(mutable_param_chain)
        return get_parameter_chain(
            mutable_param_chain,
            alt_source_attrs=alt_source_attrs,
        )

    for alt_source_attr in alt_source_attrs_set:
        alt_source = getattr(param, alt_source_attr, DOES_NOT_EXIST)
        if alt_source is None:  # Valid linking attribute, but no link parameter
            return tuple(param_chain)
        elif isinstance(alt_source, Parameter):
            mutable_param_chain.append(alt_source)
            if alt_source in param_chain:  # There is a loop in the links
                return tuple(mutable_param_chain)
            return get_parameter_chain(
                mutable_param_chain,
                alt_source_attrs=alt_source_attrs,
            )
    return tuple(param_chain)


def _merge_user_and_class_attrs(
    alt_source_attrs: str | Sequence[str] | None = None,
) -> Iterable[str]:
    """Merges user-supplied linking attributes with attributes from InferAttrs"""
    if alt_source_attrs is None:
        return InferAttrs.known_attrs()
    elif isinstance(alt_source_attrs, str):
        return set.union(set((alt_source_attrs,)), set(InferAttrs.known_attrs()))
    else:
        return set.union(set(alt_source_attrs), set(InferAttrs.known_attrs()))
