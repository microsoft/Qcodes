import warnings
from typing import Any, Optional, Sequence

import numpy as np
from numpy import VisibleDeprecationWarning


def list_of_data_to_maybe_ragged_nd_array(
    column_data: Sequence[Any],
    dtype: Optional[type] = None,
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
            # numpy warns here and coming versions
            # will eventually raise
            # for ragged arrays if you don't explicitly set
            # dtype=object
            # It is time consuming to detect ragged arrays here
            # and it is expected to be a relatively rare situation
            # so fallback to object if the regular dtype fail
            data = np.array(column_data, dtype=dtype)
    except:
        # Not clear which error to catch here. This will only be clarified
        # once numpy actually starts to raise here.
        data = np.array(column_data, dtype=object)
    return data
