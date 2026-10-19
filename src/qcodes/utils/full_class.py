from qcodes.utils._full_class import full_class
from qcodes.utils.deprecate import _warn_deprecated_module

_warn_deprecated_module(__name__, "qcodes.utils")

__all__ = ["full_class"]
