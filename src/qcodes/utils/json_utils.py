import collections
import json
import numbers
from typing import Any

import numpy as np
import uncertainties  # type: ignore[import-untyped]


class NumpyJSONEncoder(json.JSONEncoder):
    """
    This JSON encoder adds support for serializing types that the built-in
    ``json`` module does not support out-of-the-box. See the docstring of the
    ``default`` method for the description of all conversions.

    To deserialize JSON data produced by this encoder, see :class:`NumpyJSONDecoder`.
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
        elif (json_encoder := getattr(o, "_JSONEncoder", None)) is not None:
            # Use object's custom JSON encoder
            return json_encoder()
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
                if (
                    getnewargs := getattr(o, "__getnewargs__", None)
                ) is not None and not isinstance(o, (bytes, bytearray)):
                    return {
                        "__class__": type(o).__name__,
                        "__args__": getnewargs(),
                    }
                else:
                    # we cannot convert the object to JSON, just take a string
                    s = str(o)
            return s


class NumpyJSONDecoder(json.JSONDecoder):
    """
    This JSON decoder adds support for deserializing types encoded by
    :class:`NumpyJSONEncoder`, such as complex numbers and UFloat instances.

    Because Python's standard library ``json`` module does not provide a
    pluggable global registry for custom decoders, this decoder should be
    passed explicitly to :func:`json.loads` or :func:`json.load` via the
    ``cls`` argument, or by passing :meth:`default_object_hook` as the
    ``object_hook`` parameter.

    Examples:
        Decode a JSON string containing complex numbers and uncertainties::

            import json
            import uncertainties
            from qcodes.utils import NumpyJSONDecoder, NumpyJSONEncoder

            data = {
                "voltage": 1.5 + 2.0j,
                "resistance": uncertainties.ufloat(100.0, 0.5),
            }
            encoded = json.dumps(data, cls=NumpyJSONEncoder)
            decoded = json.loads(encoded, cls=NumpyJSONDecoder)

            assert decoded["voltage"] == 1.5 + 2.0j
            assert decoded["resistance"] == uncertainties.ufloat(100.0, 0.5)

        Alternatively, pass :meth:`default_object_hook` directly as ``object_hook``::

            decoded = json.loads(
                encoded, object_hook=NumpyJSONDecoder.default_object_hook
            )

        Read and decode from a JSON file using :func:`json.load`::

            with open("data.json", "r") as f:
                data = json.load(f, cls=NumpyJSONDecoder)

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if "object_hook" not in kwargs:
            kwargs["object_hook"] = self.default_object_hook
        super().__init__(*args, **kwargs)

    @staticmethod
    def default_object_hook(dct: dict[str, Any]) -> Any:
        """
        Convert dictionary representations of encoded types back to Python objects.

        If ``dct`` contains ``{"__dtype__": "complex", "re": ..., "im": ...}``,
        a :class:`complex` number is returned. If ``dct`` contains
        ``{"__dtype__": "UFloat", "nominal_value": ..., "std_dev": ...}``, an
        :class:`uncertainties.UFloat` instance is returned. Otherwise, ``dct``
        is returned unchanged.

        Args:
            dct: Dictionary parsed from JSON.

        Returns:
            The decoded object or the original dictionary if unrecognized.

        """
        dtype = dct.get("__dtype__")
        if dtype == "complex" and "re" in dct and "im" in dct:
            return complex(dct["re"], dct["im"])
        elif dtype == "UFloat" and "nominal_value" in dct and "std_dev" in dct:
            return uncertainties.ufloat(dct["nominal_value"], dct["std_dev"])
        return dct
