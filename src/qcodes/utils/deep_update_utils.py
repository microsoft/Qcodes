from collections import abc
from collections.abc import Hashable, Mapping, MutableMapping
from copy import deepcopy
from typing import TYPE_CHECKING, Any, TypeVar, cast


def deep_update[K: Hashable, L: Hashable](
    dest: MutableMapping[K, Any], update: Mapping[L, Any]
) -> MutableMapping[K | L, Any]:
    """
    Recursively update one JSON structure with another.

    Only dives into nested dicts; lists get replaced completely.
    If the original value is a dictionary and the new value is not, or vice versa,
    we also replace the value completely.
    """
    dest_int = cast("MutableMapping[K | L, Any]", dest)
    for k, v_update in update.items():
        v_dest = dest_int.get(k)
        if isinstance(v_update, abc.Mapping) and isinstance(v_dest, abc.MutableMapping):
            deep_update(v_dest, v_update)
        else:
            dest_int[k] = deepcopy(v_update)
    return dest_int


if not TYPE_CHECKING:
    from qcodes.utils.deprecate import _make_deprecated_typevars_getattr

    _deprecated_typevars: dict[str, TypeVar] = {
        "K": TypeVar("K", bound=Hashable),
        "L": TypeVar("L", bound=Hashable),
    }

    __getattr__ = _make_deprecated_typevars_getattr(__name__, _deprecated_typevars)
