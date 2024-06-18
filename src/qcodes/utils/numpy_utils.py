from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

try:
    from numpy.exceptions import VisibleDeprecationWarning
except ImportError:
    # numpy < 2.0
    from numpy import VisibleDeprecationWarning  # type: ignore[attr-defined, no-redef]


def list_of_data_to_maybe_ragged_nd_array(
    column_data: Sequence[Any],
    dtype: type | None = None,
) -> np.ndarray:
    """
    Convert a (nested) Sequence of data to numpy arrays. Handle that
    the elements of the sequence may not have the same length in which
    case the returned array will be of dtype object.

    """
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=VisibleDeprecationWarning,
                message="Creating an ndarray from ragged nested sequences",
            )
            # numpy < 1.24 warns here and coming versions
            # will eventually raise
            # for ragged arrays if you don't explicitly set
            # dtype=object
            # It is time consuming to detect ragged arrays here
            # and it is expected to be a relatively rare situation
            # so fallback to object if the regular dtype fail
            # the warning filter here can be removed once
            # we drop support for numpy < 1.24
            data = np.array(column_data, dtype=dtype)
    except ValueError:
        # From numpy 1.24 this throws a ValueError
        data = np.array(column_data, dtype=object)
    return data
