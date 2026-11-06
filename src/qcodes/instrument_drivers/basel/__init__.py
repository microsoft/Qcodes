from qcodes.utils.deprecate import _keep_attributes_shadowing_submodules

from ._basel_sp983 import BaselSP983
from ._basel_sp983a import BaselSP983a
from ._basel_sp983c import BaselSP983c

# Keep the public objects when the deprecated module aliases that share
# their names are imported.
_keep_attributes_shadowing_submodules(__name__)


__all__ = ["BaselSP983", "BaselSP983a", "BaselSP983c"]
