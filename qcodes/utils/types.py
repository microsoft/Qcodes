"""
Useful collections of types used around QCoDeS
"""


import numpy as np
from typing import Union

numpy_concrete_ints = (np.int8, np.int16, np.int32, np.int64,
                       np.uint8, np.uint16, np.uint32, np.uint64)
numpy_non_concrete_ints_instantiable = (np.int, np.int_, np.uint)
numpy_non_concrete_ints = \
    numpy_non_concrete_ints_instantiable + (np.integer,)

numpy_concrete_floats = (np.float16, np.float32, np.float64)
numpy_non_concrete_floats_instantiable = (np.float, np.float_)
numpy_non_concrete_floats = \
    numpy_non_concrete_floats_instantiable + (np.floating,)

numpy_concrete_complex = (np.complex64, np.complex128)
numpy_non_concrete_complex = (np.complex, np.complex_, np.complexfloating)


complex_types = numpy_concrete_complex + numpy_non_concrete_complex + (complex,)

# These are the same types as a above unfortunately there does not seem to be
# a good way to convert a tuple of types to a Union
complex_type_union = Union[np.complex64, np.complex128, np.complex,
                           np.complex_, np.complexfloating, complex]
