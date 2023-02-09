"""
Module left for backwards compatibility.
Please do not import from this in any new code
"""
import logging
from typing import Any, Dict, Hashable, Optional, Tuple

# for backwards compatibility since this module used
# to contain logic that would abstract between yaml
# libraries.
from ruamel.yaml import YAML

from qcodes.parameters.named_repr import named_repr
from qcodes.parameters.permissive_range import permissive_range
from qcodes.parameters.sequence_helpers import is_sequence, is_sequence_of
from qcodes.parameters.sweep_values import make_sweep
from qcodes.parameters.val_mapping import create_on_off_val_mapping
from qcodes.utils.deprecate import deprecate

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
from .spyder_utils import add_to_spyder_UMR_excludelist


# on longer in used but left for backwards compatibility until
# module is removed.
def warn_units(class_name: str, instance: object) -> None:
    logging.warning('`units` is deprecated for the `' + class_name +
                    '` class, use `unit` instead. ' + repr(instance))


@deprecate("Internal function no longer part of the public qcodes api")
def compare_dictionaries(
    dict_1: Dict[Hashable, Any],
    dict_2: Dict[Hashable, Any],
    dict_1_name: Optional[str] = "d1",
    dict_2_name: Optional[str] = "d2",
    path: str = "",
) -> Tuple[bool, str]:
    """
    Compare two dictionaries recursively to find non matching elements.

    Args:
        dict_1: First dictionary to compare.
        dict_2: Second dictionary to compare.
        dict_1_name: Optional name of the first dictionary used in the
                     differences string.
        dict_2_name: Optional name of the second dictionary used in the
                     differences string.
    Returns:
        Tuple: Are the dicts equal and the difference rendered as
               a string.

    """
    err = ""
    key_err = ""
    value_err = ""
    old_path = path
    for k in dict_1.keys():
        path = old_path + "[%s]" % k
        if k not in dict_2.keys():
            key_err += f"Key {dict_1_name}{path} not in {dict_2_name}\n"
        else:
            if isinstance(dict_1[k], dict) and isinstance(dict_2[k], dict):
                err += compare_dictionaries(
                    dict_1[k], dict_2[k], dict_1_name, dict_2_name, path
                )[1]
            else:
                match = dict_1[k] == dict_2[k]

                # if values are equal-length numpy arrays, the result of
                # "==" is a bool array, so we need to 'all' it.
                # In any other case "==" returns a bool
                # TODO(alexcjohnson): actually, if *one* is a numpy array
                # and the other is another sequence with the same entries,
                # this will compare them as equal. Do we want this, or should
                # we require exact type match?
                if hasattr(match, "all"):
                    match = match.all()

                if not match:
                    value_err += (
                        'Value of "{}{}" ("{}", type"{}") not same as\n'
                        '  "{}{}" ("{}", type"{}")\n\n'
                    ).format(
                        dict_1_name,
                        path,
                        dict_1[k],
                        type(dict_1[k]),
                        dict_2_name,
                        path,
                        dict_2[k],
                        type(dict_2[k]),
                    )

    for k in dict_2.keys():
        path = old_path + f"[{k}]"
        if k not in dict_1.keys():
            key_err += f"Key {dict_2_name}{path} not in {dict_1_name}\n"

    dict_differences = key_err + value_err + err
    if len(dict_differences) == 0:
        dicts_equal = True
    else:
        dicts_equal = False
    return dicts_equal, dict_differences
