"""
Useful collections of types used around QCoDeS
"""


import numpy as np

numpy_concrete_ints = (np.int8, np.int16, np.int32, np.int64,
                       np.uint8, np.uint16, np.uint32, np.uint64)
numpy_non_concrete_ints = (np.int, np.int_, np.uint, np.integer)

numpy_concrete_floats = (np.float16, np.float32, np.float64)
numpy_non_concrete_floats = (np.float, np.float_,
                             np.floating)

numpy_concrete_complex = (np.complex64, np.complex128)
numpy_non_concrete_complex = (np.complex, np.complex_, np.complexfloating)


complex_types = numpy_concrete_complex + numpy_non_concrete_complex + (complex,)