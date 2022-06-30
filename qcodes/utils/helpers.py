"""
Module left for backwards compatibility.
Please do not import from this in any new code
"""
import logging

# for backwards compatibility since this module used
# to contain logic that would abstract between yaml
# libraries.
from ruamel.yaml import YAML

from qcodes.loops import tprint, wait_secs
from qcodes.parameters.named_repr import named_repr
from qcodes.parameters.permissive_range import permissive_range
from qcodes.parameters.sequence_helpers import is_sequence, is_sequence_of
from qcodes.parameters.sweep_values import make_sweep
from qcodes.tests.common import compare_dictionaries

from .abstractmethod import qcodes_abstractmethod as abstractmethod
from .attribute_helpers import (
    DelegateAttributes,
    attribute_set_to,
    checked_getattr,
    strip_attrs,
)
from .deep_update_utils import deep_update
from .full_class import full_class
from .function_helpers import is_function
from .json_utils import NumpyJSONEncoder
from .partial_utils import partial_with_docstring
from .path_helpers import QCODES_USER_PATH_ENV, get_qcodes_path, get_qcodes_user_path
from .qt_helpers import foreground_qt_window
from .spyder_utils import add_to_spyder_UMR_excludelist
from .val_mapping import create_on_off_val_mapping


# on longer in used but left for backwards compatibility until
# module is removed.
def warn_units(class_name: str, instance: object) -> None:
    logging.warning('`units` is deprecated for the `' + class_name +
                    '` class, use `unit` instead. ' + repr(instance))
