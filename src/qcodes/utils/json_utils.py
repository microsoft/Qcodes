import collections
import json
import numbers
from typing import Any

import numpy as np


class NumpyJSONEncoder(json.JSONEncoder):
    """
    This JSON encoder adds support for serializing types that the built-in
    ``json`` module does not support out-of-the-box. See the docstring of the
    ``default`` method for the description of all conversions.
    """

    def default(self, o: Any) -> Any:
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
        * Numbers with uncertainties  (numbers that conforms to
          ``uncertainties.UFloat``) get converted to a dictionary with fields
          ``nominal_value`` and ``std_dev`` containing floating numbers for the
          nominal and uncertainty parts respectively, and a field
          ``__dtype__`` containing value ``UFloat``.
        * Object with a ``_JSONEncoder`` method get converted the return value of
          that method.
        * Objects which support the pickle protocol get converted using the
          data provided by that protocol.
        * Other objects which cannot be serialized get converted to their
          string representation (using the ``str`` function).
        """
        import uncertainties  # type: ignore[import-untyped]

        if isinstance(o, np.generic) and not isinstance(o, np.complexfloating):
            # for numpy scalars
            return o.item()
        elif isinstance(o, np.ndarray):
            # for numpy arrays
            return o.tolist()
        elif isinstance(o, numbers.Complex) and not isinstance(o, numbers.Real):
            return {
                "__dtype__": "complex",
                "re": float(o.real),
                "im": float(o.imag),
            }
        elif isinstance(o, uncertainties.UFloat):
            return {
                "__dtype__": "UFloat",
                "nominal_value": float(o.nominal_value),
                "std_dev": float(o.std_dev),
            }
        elif hasattr(o, "_JSONEncoder"):
            # Use object's custom JSON encoder
            jsosencode = getattr(o, "_JSONEncoder")
            return jsosencode()
        else:
            try:
                s = super().default(o)
            except TypeError:
                # json does not support dumping UserDict but
                # we can dump the dict stored internally in the
                # UserDict
                if isinstance(o, collections.UserDict):
                    return o.data
                # See if the object supports the pickle protocol.
                # If so, we should be able to use that to serialize.
                # __getnewargs__ will return bytes for a bytes object
                # causing an infinte recursion, so we do not
                # try to pickle bytes or bytearrays
                if hasattr(o, "__getnewargs__") and not isinstance(
                    o, (bytes, bytearray)
                ):
                    return {
                        "__class__": type(o).__name__,
                        "__args__": getattr(o, "__getnewargs__")(),
                    }
                else:
                    # we cannot convert the object to JSON, just take a string
                    s = str(o)
            return s
