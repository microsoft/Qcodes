import collections
import json
import numbers
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


class NumpyJSONEncoder(json.JSONEncoder):
    """
    This JSON encoder adds support for serializing types that the built-in
    ``json`` module does not support out-of-the-box. See the docstring of the
    ``default`` method for the description of all conversions.
    """

    def default(self, obj: Any) -> Any:
        """
        List of conversions that this encoder performs:

        * ``numpy.generic`` (all integer, floating, and other types) gets
          converted to its python equivalent using its ``item`` method (see
          ``numpy`` docs for more information,
          https://docs.scipy.org/doc/numpy/reference/arrays.scalars.html).
        * ``numpy.ndarray`` gets converted to python list using its ``tolist``
          method.
        * Complex number (a number that conforms to ``numbers.Complex`` ABC) gets
          converted to a dictionary with fields ``re`` and ``im`` containing floating
          numbers for the real and imaginary parts respectively, and a field
          ``__dtype__`` containing value ``complex``.
        * Numbers with uncertainties  (numbers that conforms to ``uncertainties.UFloat``) get
          converted to a dictionary with fields ``nominal_value`` and ``std_dev`` containing floating
          numbers for the nominal and uncertainty parts respectively, and a field
          ``__dtype__`` containing value ``UFloat``.
        * Object with a ``_JSONEncoder`` method get converted the return value of
          that method.
        * Objects which support the pickle protocol get converted using the
          data provided by that protocol.
        * Other objects which cannot be serialized get converted to their
          string representation (using the ``str`` function).
        """
        with warnings.catch_warnings():
            # this context manager can be removed when uncertainties
            # no longer triggers deprecation warnings
            warnings.simplefilter("ignore", category=DeprecationWarning)
            import uncertainties

        if isinstance(obj, np.generic) and not isinstance(obj, np.complexfloating):
            # for numpy scalars
            return obj.item()
        elif isinstance(obj, np.ndarray):
            # for numpy arrays
            return obj.tolist()
        elif isinstance(obj, numbers.Complex) and not isinstance(obj, numbers.Real):
            return {
                "__dtype__": "complex",
                "re": float(obj.real),
                "im": float(obj.imag),
            }
        elif isinstance(obj, uncertainties.UFloat):
            return {
                "__dtype__": "UFloat",
                "nominal_value": float(obj.nominal_value),
                "std_dev": float(obj.std_dev),
            }
        elif hasattr(obj, "_JSONEncoder"):
            # Use object's custom JSON encoder
            jsosencode = getattr(obj, "_JSONEncoder")
            return jsosencode()
        else:
            try:
                s = super().default(obj)
            except TypeError:
                # json does not support dumping UserDict but
                # we can dump the dict stored internally in the
                # UserDict
                if isinstance(obj, collections.UserDict):
                    return obj.data
                # See if the object supports the pickle protocol.
                # If so, we should be able to use that to serialize.
                if hasattr(obj, "__getnewargs__"):
                    return {
                        "__class__": type(obj).__name__,
                        "__args__": getattr(obj, "__getnewargs__")(),
                    }
                else:
                    # we cannot convert the object to JSON, just take a string
                    s = str(obj)
            return s
