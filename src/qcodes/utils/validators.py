"""
Backwards compatible. New code should import from qcodes.validators
"""

import warnings

from qcodes.utils.deprecate import QCoDeSDeprecationWarning
from qcodes.validators import (
    Anything,
    Arrays,
    Bool,
    Callable,
    ComplexNumbers,
    Dict,
    Enum,
    Ints,
    Lists,
    Multiples,
    MultiType,
    MultiTypeAnd,
    MultiTypeOr,
    Nothing,
    Numbers,
    OnOff,
    PermissiveInts,
    PermissiveMultiples,
    Sequence,
    Strings,
    Validator,
    validate_all,
)
from qcodes.validators.validators import (
    BIGINT,
    BIGSTRING,
    T,
    numbertypes,
    range_str,
    shape_tuple_type,
    shape_type,
)

warnings.warn(
    "The `qcodes.utils.validators` module is deprecated. "
    "Please consult the api documentation at https://microsoft.github.io/Qcodes/api/index.html for alternatives.",
    category=QCoDeSDeprecationWarning,
    stacklevel=2,
)
