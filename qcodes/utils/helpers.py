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

# from qcodes.loops import tprint
# from qcodes.parameters.sequence_helpers import is_sequence, is_sequence_of
# from qcodes.parameters.permissive_range import permissive_range
# from qcodes.tests.common import compare_dictionaries
# from qcodes.parameters.sweep_values import make_sweep
# from qcodes.parameters.named_repr import named_repr
from .full_class import full_class
from .function_helpers import is_function
from .json_utils import NumpyJSONEncoder
from .path_helpers import QCODES_USER_PATH_ENV, get_qcodes_path, get_qcodes_user_path
from .val_mapping import create_on_off_val_mapping

if TYPE_CHECKING:
    from PyQt5.QtWidgets import QMainWindow


log = logging.getLogger(__name__)


K = TypeVar('K', bound=Hashable)
L = TypeVar('L', bound=Hashable)


def deep_update(
        dest: MutableMapping[K, Any],
        update: Mapping[L, Any]
) -> MutableMapping[Union[K, L], Any]:
    """
    Recursively update one JSON structure with another.

    Only dives into nested dicts; lists get replaced completely.
    If the original value is a dictionary and the new value is not, or vice versa,
    we also replace the value completely.
    """
    dest_int = cast(MutableMapping[Union[K, L], Any], dest)
    for k, v_update in update.items():
        v_dest = dest_int.get(k)
        if isinstance(v_update, abc.Mapping) and isinstance(v_dest, abc.MutableMapping):
            deep_update(v_dest, v_update)
        else:
            dest_int[k] = deepcopy(v_update)
    return dest_int


def wait_secs(finish_clock: float) -> float:
    """
    Calculate the number of seconds until a given clock time.
    The clock time should be the result of ``time.perf_counter()``.
    Does NOT wait for this time.
    """
    delay = finish_clock - time.perf_counter()
    if delay < 0:
        logging.warning(f'negative delay {delay:.6f} sec')
        return 0
    return delay


def warn_units(class_name: str, instance: object) -> None:
    logging.warning('`units` is deprecated for the `' + class_name +
                    '` class, use `unit` instead. ' + repr(instance))


def foreground_qt_window(window: "QMainWindow") -> None:
    """
    Try as hard as possible to bring a qt window to the front. This
    will use pywin32 if installed and running on windows as this
    seems to be the only reliable way to foreground a window. The
    build-in qt functions often doesn't work. Note that to use this
    with pyqtgraphs remote process you should use the ref in that module
    as in the example below.

    Args:
        window: Handle to qt window to foreground.
    Examples:
        >>> Qtplot.qt_helpers.foreground_qt_window(plot.win)
    """
    try:
        import win32con
        from win32gui import SetWindowPos

        # use the idea from
        # https://stackoverflow.com/questions/12118939/how-to-make-a-pyqt4-window-jump-to-the-front
        SetWindowPos(window.winId(),
                     win32con.HWND_TOPMOST, # = always on top. only reliable way to bring it to the front on windows
                     0, 0, 0, 0,
                     win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_SHOWWINDOW)
        SetWindowPos(window.winId(),
                     win32con.HWND_NOTOPMOST, # disable the always on top, but leave window at its top position
                     0, 0, 0, 0,
                     win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_SHOWWINDOW)
    except ImportError:
        pass
    window.show()
    window.raise_()
    window.activateWindow()


def add_to_spyder_UMR_excludelist(modulename: str) -> None:
    """
    Spyder tries to reload any user module. This does not work well for
    qcodes because it overwrites Class variables. QCoDeS uses these to
    store global attributes such as default station, monitor and list of
    instruments. This "feature" can be disabled by the
    gui. Unfortunately this cannot be disabled in a natural way
    programmatically so in this hack we replace the global ``__umr__`` instance
    with a new one containing the module we want to exclude. This will do
    nothing if Spyder is not found.
    TODO is there a better way to detect if we are in spyder?
    """
    if any('SPYDER' in name for name in os.environ):

        sitecustomize_found = False
        try:
            from spyder.utils.site import sitecustomize
        except ImportError:
            pass
        else:
            sitecustomize_found = True
        if sitecustomize_found is False:
            try:
                from spyder_kernels.customize import spydercustomize as sitecustomize

            except ImportError:
                pass
            else:
                sitecustomize_found = True

        if sitecustomize_found is False:
            return

        excludednamelist = os.environ.get('SPY_UMR_NAMELIST',
                                          '').split(',')
        if modulename not in excludednamelist:
            log.info(f"adding {modulename} to excluded modules")
            excludednamelist.append(modulename)
            sitecustomize.__umr__ = sitecustomize.UserModuleReloader(namelist=excludednamelist)
            os.environ['SPY_UMR_NAMELIST'] = ','.join(excludednamelist)


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


def partial_with_docstring(func: Callable[..., Any],
                           docstring: str,
                           **kwargs: Any) -> Callable[..., Any]:
    """
    We want to have a partial function which will allow us to access the docstring
    through the python built-in help function. This is particularly important
    for client-facing driver methods, whose arguments might not be obvious.

    Consider the follow example why this is needed:

    >>> from functools import partial
    >>> def f():
    >>> ... pass
    >>> g = partial(f)
    >>> g.__doc__ = "bla"
    >>> help(g) # this will print the docstring of partial and not the docstring set above

    Args:
        func: A function that its docstring will be accessed.
        docstring: The docstring of the corresponding function.
    """
    ex = partial(func, **kwargs)

    def inner(*inner_args: Any, **inner_kwargs: Any) -> Any:
        return ex(*inner_args, **inner_kwargs)

    inner.__doc__ = docstring

    return inner
