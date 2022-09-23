from __future__ import annotations

from abc import ABCMeta
from typing import Any


class InstrumentMeta(ABCMeta):
    """
    Metaclass used to customize Instrument creation. We want to register the
    instance iff __init__ successfully runs, however we can only do this if
    we customize the instance initialization process, otherwise there is no
    way to run `register_instance` after `__init__` but before the created
    instance is returned to the caller.

    Instead we use the fact that `__new__` and `__init__` are called inside
    `type.__call__`
    (https://github.com/python/cpython/blob/main/Objects/typeobject.c#L1077,
    https://github.com/python/typeshed/blob/master/stdlib/builtins.pyi#L156)
    which we will overload to insert our own custom code AFTER `__init__` is
    complete. Note this is part of the spec and will work in alternate python
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
        Overloads `type.__call__` to add code that runs only if __init__ completes
        successfully.
        """
        new_inst = super().__call__(*args, **kwargs)
        is_abstract = new_inst._is_abstract()
        if is_abstract:
            new_inst.close()
            raise NotImplementedError(
                f"{new_inst} has un-implemented Abstract Parameter "
                f"and cannot be initialized"
            )

        new_inst.record_instance(new_inst)
        return new_inst
