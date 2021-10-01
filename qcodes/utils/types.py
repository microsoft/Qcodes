"""
Useful collections of types used around QCoDeS
"""
from typing import Tuple, Union

import numpy as np

numpy_concrete_ints = (np.int8, np.int16, np.int32, np.int64,
                       np.uint8, np.uint16, np.uint32, np.uint64)
"""
Integer types with fixed sizes.
"""
numpy_c_ints = (np.uintp, np.uintc, np.intp, np.intc,
                np.short, np.byte, np.ushort, np.ubyte,
                np.longlong, np.ulonglong)
"""
Integer types that matches C types.
"""
numpy_non_concrete_ints_instantiable = (np.int_, np.uint)
"""
Default integer types. The size may be platform dependent.
"""

numpy_ints: Tuple[type, ...] = (
        numpy_concrete_ints +
        numpy_c_ints +
        numpy_non_concrete_ints_instantiable
)
"""
All numpy integer types
"""


numpy_concrete_floats = (np.float16, np.float32, np.float64)
"""
Floating point types with fixed sizes.
"""
numpy_c_floats = (np.half, np.single, np.double)
"""
Floating point types that matches C types.
"""
numpy_non_concrete_floats_instantiable = (np.float_, )
"""
Default floating point types. The size may be platform dependent.
"""

numpy_floats: Tuple[type, ...] = (
        numpy_concrete_floats +
        numpy_c_floats +
        numpy_non_concrete_floats_instantiable
)
"""
All numpy float types
"""

numpy_concrete_complex = (np.complex64, np.complex128)
"""
Complex types with fixed sizes.
"""
numpy_c_complex = (np.csingle, np.cdouble)
"""
Complex types that matches C types.
"""
numpy_non_concrete_complex_instantiable = (np.complex_, )
"""
Default complex types. The size may be platform dependent.
"""

numpy_complex = (
        numpy_concrete_complex +
        numpy_c_complex +
        numpy_non_concrete_complex_instantiable
)
"""
All numpy complex types
"""

concrete_complex_types = numpy_concrete_complex + (complex,)
complex_types = (
        numpy_concrete_complex +
        numpy_non_concrete_complex_instantiable +
        (complex,)
)

# These are the same types as a above unfortunately there does not seem to be
# a good way to convert a tuple of types to a Union
complex_type_union = Union[np.complex64, np.complex128,
                           np.complex_, np.complexfloating, complex]
