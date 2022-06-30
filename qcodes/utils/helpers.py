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
from .json_utils import NumpyJSONEncoder
from .path_helpers import QCODES_USER_PATH_ENV, get_qcodes_path, get_qcodes_user_path
from .val_mapping import create_on_off_val_mapping

# from qcodes.loops import tprint
# from qcodes.parameters.sequence_helpers import is_sequence, is_sequence_of
# from qcodes.parameters.permissive_range import permissive_range
# from qcodes.tests.common import compare_dictionaries

if TYPE_CHECKING:
    from PyQt5.QtWidgets import QMainWindow


log = logging.getLogger(__name__)


def is_function(f: object, arg_count: int, coroutine: bool = False) -> bool:
    """
    Check and require a function that can accept the specified number of
    positional arguments, which either is or is not a coroutine
    type casting "functions" are allowed, but only in the 1-argument form.

    Args:
        f: Function to check.
        arg_count: Number of argument f should accept.
        coroutine: Is a coroutine.

    Return:
        bool: is function and accepts the specified number of arguments.

    """
    if not isinstance(arg_count, int) or arg_count < 0:
        raise TypeError('arg_count must be a non-negative integer')

    if not (callable(f) and bool(coroutine) is iscoroutinefunction(f)):
        return False

    if isinstance(f, type):
        # for type casting functions, eg int, str, float
        # only support the one-parameter form of these,
        # otherwise the user should make an explicit function.
        return arg_count == 1

    try:
        sig = signature(f)
    except ValueError:
        # some built-in functions/methods don't describe themselves to inspect
        # we already know it's a callable and coroutine is correct.
        return True

    try:
        inputs = [0] * arg_count
        sig.bind(*inputs)
        return True
    except TypeError:
        return False


def full_class(obj: object) -> str:
    """The full importable path to an object's class."""
    return type(obj).__module__ + '.' + type(obj).__name__


def named_repr(obj: Any) -> str:
    """Enhance the standard repr() with the object's name attribute."""
    s = '<{}.{}: {} at {}>'.format(
        obj.__module__,
        type(obj).__name__,
        str(obj.name),
        id(obj))
    return s


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


# This is very much related to the permissive_range but more
# strict on the input, start and endpoints are always included,
# and a sweep is only created if the step matches an integer
# number of points.
# numpy is a dependency anyways.
# Furthermore the sweep allows to take a number of points and generates
# an array with endpoints included, which is more intuitive to use in a sweep.
def make_sweep(start: float,
               stop: float,
               step: Optional[float] = None,
               num: Optional[int] = None
               ) -> List[float]:
    """
    Generate numbers over a specified interval.
    Requires ``start`` and ``stop`` and (``step`` or ``num``).
    The sign of ``step`` is not relevant.

    Args:
        start: The starting value of the sequence.
        stop: The end value of the sequence.
        step:  Spacing between values.
        num: Number of values to generate.

    Returns:
        numpy.ndarray: numbers over a specified interval as a ``numpy.linspace``.

    Examples:
        >>> make_sweep(0, 10, num=5)
        [0.0, 2.5, 5.0, 7.5, 10.0]
        >>> make_sweep(5, 10, step=1)
        [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        >>> make_sweep(15, 10.5, step=1.5)
        >[15.0, 13.5, 12.0, 10.5]
    """
    if step and num:
        raise AttributeError('Don\'t use `step` and `num` at the same time.')
    if (step is None) and (num is None):
        raise ValueError('If you really want to go from `start` to '
                         '`stop` in one step, specify `num=2`.')
    if step is not None:
        steps = abs((stop - start) / step)
        tolerance = 1e-10
        steps_lo = int(np.floor(steps + tolerance))
        steps_hi = int(np.ceil(steps - tolerance))

        if steps_lo != steps_hi:
            raise ValueError(
                'Could not find an integer number of points for '
                'the the given `start`, `stop`, and `step` '
                'values. \nNumber of points is {:d} or {:d}.'
                .format(steps_lo + 1, steps_hi + 1))
        num_steps = steps_lo + 1
    elif num is not None:
        num_steps = num

    output_list = np.linspace(start, stop, num=num_steps).tolist()
    return cast(List[float], output_list)


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
