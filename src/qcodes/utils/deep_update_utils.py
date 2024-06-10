from collections import abc
from collections.abc import Hashable, Mapping, MutableMapping
from copy import deepcopy
from typing import Any, TypeVar, cast

K = TypeVar("K", bound=Hashable)
L = TypeVar("L", bound=Hashable)


def deep_update(
    dest: MutableMapping[K, Any], update: Mapping[L, Any]
) -> MutableMapping[K | L, Any]:
    """
    Recursively update one JSON structure with another.

    Only dives into nested dicts; lists get replaced completely.
    If the original value is a dictionary and the new value is not, or vice versa,
    we also replace the value completely.
    """
    dest_int = cast(MutableMapping[K | L, Any], dest)
    for k, v_update in update.items():
        v_dest = dest_int.get(k)
        if isinstance(v_update, abc.Mapping) and isinstance(v_dest, abc.MutableMapping):
            deep_update(v_dest, v_update)
        else:
            dest_int[k] = deepcopy(v_update)
    return dest_int
