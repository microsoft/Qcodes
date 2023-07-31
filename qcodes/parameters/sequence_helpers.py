from __future__ import annotations

import io
from collections.abc import Iterator, Sequence
from typing import Any, cast

import numpy as np


def is_sequence(obj: Any) -> bool:
    """
    Test if an object is a sequence.

    We do not consider strings or unordered collections like sets to be
    sequences, but we do accept iterators (such as generators).
    """
    return isinstance(obj, (Iterator, Sequence, np.ndarray)) and not isinstance(
        obj, (str, bytes, io.IOBase)
    )


def is_sequence_of(
    obj: Any,
    types: type[object | None] | tuple[type[object | None], ...] | None = None,
    depth: int | None = None,
    shape: Sequence[int] | None = None,
) -> bool:
    """
    Test if object is a sequence of entirely certain class(es).

    Args:
        obj: The object to test.
        types: Allowed type(s). If omitted, we just test the depth/shape.
        depth: Level of nesting, ie if ``depth=2`` we expect a sequence of
               sequences. Default 1 unless ``shape`` is supplied.
        shape: The shape of the sequence, ie its length in each dimension.
               If ``depth`` is omitted, but ``shape`` included, we set
               ``depth = len(shape)``.

    Returns:
        bool: ``True`` if every item in ``obj`` matches ``types``.
    """
    if not is_sequence(obj):
        return False

    if shape is None or shape == ():
        next_shape: tuple[int, ...] | None = None
        if depth is None:
            depth = 1
    else:
        if depth is None:
            depth = len(shape)
        elif depth != len(shape):
            raise ValueError("inconsistent depth and shape")

        if len(obj) != shape[0]:
            return False

        next_shape = cast(tuple[int, ...], shape[1:])

    for item in obj:
        if depth > 1:
            if not is_sequence_of(item, types, depth=depth - 1, shape=next_shape):
                return False
        elif types is not None and not isinstance(item, types):
            return False
    return True
