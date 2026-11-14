from __future__ import annotations

from abc import ABCMeta
from typing import Any


class InstrumentMeta(ABCMeta):
    """
    Metaclass used to customize Instrument creation. We want to register the
    instance iff __init__ successfully runs, and close it right away if
    __init__ or the registration fails, however we can only do this if we
    customize the instance initialization process, otherwise there is no way
    to run code after `__init__` but before the created instance is returned
    to the caller, or to get hold of the instance if `__init__` raises.

    Instead we use the fact that `__new__` and `__init__` are called inside
    `type.__call__`
    (https://github.com/python/cpython/blob/main/Objects/typeobject.c#L1077,
    https://github.com/python/typeshed/blob/master/stdlib/builtins.pyi#L156)
    which we will overload to insert our own custom code around `__init__`.
    Note this is part of the spec and will work in alternate python
    implementations like pypy too.

    We inherit from ABCMeta rather than type for backwards compatibility
    reasons. There may be instrument interfaces that are defined in
    terms of an ABC. Inheriting directly from type here would then give
    `TypeError: metaclass conflict: the metaclass of a derived class must
    be a (non-strict) subclass of the metaclasses of all its bases`
    for a class that inherits from ABC
    """

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        """
        Overloads `type.__call__` to add code that runs only if `__init__`
        completes successfully, and to close the instance if `__init__` or that
        code raises.
        """
        # Create and initialize the instance like type.__call__ does.
        new_inst: Any = cls.__new__(cls, *args, **kwargs)
        if not issubclass(type(new_inst), cls):
            return new_inst
        try:
            result = type(new_inst).__init__(new_inst, *args, **kwargs)
            if result is not None:
                raise TypeError(
                    f"__init__() should return None, not '{type(result).__name__}'"
                )
            if new_inst._is_abstract():
                raise NotImplementedError(
                    f"{new_inst} has un-implemented Abstract Parameter "
                    f"and cannot be initialized"
                )
            new_inst.record_instance(new_inst)
        except BaseException as error:
            # Free the resources that the instance acquired before the failure
            # right away, rather than when it is garbage collected, since the
            # caller never gets hold of it.
            new_inst._close_after_failed_initialization(error)
            raise
        return new_inst
