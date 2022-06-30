import io
import logging
import math
import os
import time
from asyncio import iscoroutinefunction
from collections import OrderedDict, abc
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
from inspect import signature
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Hashable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    SupportsAbs,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import numpy as np

# for backwards compatibility since this module used
# to contain logic that would abstract between yaml
# libraries.
from ruamel.yaml import YAML

from .abstractmethod import qcodes_abstractmethod as abstractmethod
from .attribute_helpers import DelegateAttributes, checked_getattr, strip_attrs
from .deep_update_utils import deep_update

# from qcodes.loops import tprint
# from qcodes.parameters.sequence_helpers import is_sequence, is_sequence_of
# from qcodes.parameters.permissive_range import permissive_range
# from qcodes.tests.common import compare_dictionaries
# from qcodes.parameters.sweep_values import make_sweep
# from qcodes.parameters.named_repr import named_repr
# from qcodes.loops import wait_secs
from .full_class import full_class
from .function_helpers import is_function
from .json_utils import NumpyJSONEncoder
from .partial_utils import partial_with_docstring
from .path_helpers import QCODES_USER_PATH_ENV, get_qcodes_path, get_qcodes_user_path
from .qt_helpers import foreground_qt_window
from .spyder_utils import add_to_spyder_UMR_excludelist
from .val_mapping import create_on_off_val_mapping

log = logging.getLogger(__name__)


def warn_units(class_name: str, instance: object) -> None:
    logging.warning('`units` is deprecated for the `' + class_name +
                    '` class, use `unit` instead. ' + repr(instance))


@contextmanager
def attribute_set_to(object_: object,
                     attribute_name: str,
                     new_value: Any) -> Iterator[None]:
    """
    This context manager allows to change a given attribute of a given object
    to a new value, and the original value is reverted upon exit of the context
    manager.

    Args:
        object_: The object which attribute value is to be changed.
        attribute_name: The name of the attribute that is to be changed.
        new_value: The new value to which the attribute of the object is
                   to be changed.
    """
    old_value = getattr(object_, attribute_name)
    setattr(object_, attribute_name, new_value)
    try:
        yield
    finally:
        setattr(object_, attribute_name, old_value)
