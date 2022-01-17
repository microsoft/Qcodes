import collections
import io
import json
import logging
import math
import numbers
import os
import time
import warnings
from asyncio import iscoroutinefunction
from collections import OrderedDict, abc
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
from inspect import signature
from pathlib import Path
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

if TYPE_CHECKING:
    from PyQt5.QtWidgets import QMainWindow

QCODES_USER_PATH_ENV = 'QCODES_USER_PATH'


_tprint_times: Dict[str, float] = {}


log = logging.getLogger(__name__)


class NumpyJSONEncoder(json.JSONEncoder):
    """
    This JSON encoder adds support for serializing types that the built-in
    ``json`` module does not support out-of-the-box. See the docstring of the
    ``default`` method for the description of all conversions.
    """

    def default(self, obj: Any) -> Any:
        """
        List of conversions that this encoder performs:

        * ``numpy.generic`` (all integer, floating, and other types) gets
          converted to its python equivalent using its ``item`` method (see
          ``numpy`` docs for more information,
          https://docs.scipy.org/doc/numpy/reference/arrays.scalars.html).
        * ``numpy.ndarray`` gets converted to python list using its ``tolist``
          method.
        * Complex number (a number that conforms to ``numbers.Complex`` ABC) gets
          converted to a dictionary with fields ``re`` and ``im`` containing floating
          numbers for the real and imaginary parts respectively, and a field
          ``__dtype__`` containing value ``complex``.
        * Numbers with uncertainties  (numbers that conforms to ``uncertainties.UFloat``) get
          converted to a dictionary with fields ``nominal_value`` and ``std_dev`` containing floating
          numbers for the nominal and uncertainty parts respectively, and a field
          ``__dtype__`` containing value ``UFloat``.
        * Object with a ``_JSONEncoder`` method get converted the return value of
          that method.
        * Objects which support the pickle protocol get converted using the
          data provided by that protocol.
        * Other objects which cannot be serialized get converted to their
          string representation (using the ``str`` function).
        """
        with warnings.catch_warnings():
            # this context manager can be removed when uncertainties
            # no longer triggers deprecation warnings
            warnings.simplefilter("ignore", category=DeprecationWarning)
            import uncertainties

        if isinstance(obj, np.generic) \
                and not isinstance(obj, np.complexfloating):
            # for numpy scalars
            return obj.item()
        elif isinstance(obj, np.ndarray):
            # for numpy arrays
            return obj.tolist()
        elif (isinstance(obj, numbers.Complex) and
              not isinstance(obj, numbers.Real)):
            return {
                '__dtype__': 'complex',
                're': float(obj.real),
                'im': float(obj.imag)
            }
        elif isinstance(obj, uncertainties.UFloat):
            return {
                '__dtype__': 'UFloat',
                'nominal_value': float(obj.nominal_value),
                'std_dev': float(obj.std_dev)
            }
        elif hasattr(obj, '_JSONEncoder'):
            # Use object's custom JSON encoder
            jsosencode = getattr(obj, "_JSONEncoder")
            return jsosencode()
        else:
            try:
                s = super().default(obj)
            except TypeError:
                # json does not support dumping UserDict but
                # we can dump the dict stored internally in the
                # UserDict
                if isinstance(obj, collections.UserDict):
                    return obj.data
                # See if the object supports the pickle protocol.
                # If so, we should be able to use that to serialize.
                if hasattr(obj, '__getnewargs__'):
                    return {
                        '__class__': type(obj).__name__,
                        '__args__': getattr(obj, "__getnewargs__")()
                    }
                else:
                    # we cannot convert the object to JSON, just take a string
                    s = str(obj)
            return s


def tprint(string: str, dt: int = 1, tag: str = 'default') -> None:
    """Print progress of a loop every ``dt`` seconds."""
    ptime = _tprint_times.get(tag, 0)
    if (time.time() - ptime) > dt:
        print(string)
        _tprint_times[tag] = time.time()


def is_sequence(obj: Any) -> bool:
    """
    Test if an object is a sequence.

    We do not consider strings or unordered collections like sets to be
    sequences, but we do accept iterators (such as generators).
    """
    return (isinstance(obj, (abc.Iterator, abc.Sequence, np.ndarray)) and
            not isinstance(obj, (str, bytes, io.IOBase)))


def is_sequence_of(obj: Any,
                   types: Optional[Union[Type[object],
                                         Tuple[Type[object], ...]]] = None,
                   depth: Optional[int] = None,
                   shape: Optional[Sequence[int]] = None
                   ) -> bool:
    """
    Test if object is a sequence of entirely certain class(es).

    Args:
        obj: The object to test.
        types: Allowed type(s). If omitted, we just test the depth/shape.
        depth: Level of nesting, ie if ``depth=2`` we expect a sequence of
               sequences. Default 1 unless ``shape`` is supplied.
        shape: The shape of the sequence, ie its length in each dimension.
               If ``depth`` is omitted, but ``shape`` included, we set
               ``depth = len(shape)``.

    Returns:
        bool: ``True`` if every item in ``obj`` matches ``types``.
    """
    if not is_sequence(obj):
        return False

    if shape is None or shape == ():
        next_shape: Optional[Tuple[int]] = None
        if depth is None:
            depth = 1
    else:
        if depth is None:
            depth = len(shape)
        elif depth != len(shape):
            raise ValueError('inconsistent depth and shape')

        if len(obj) != shape[0]:
            return False

        next_shape = cast(Tuple[int], shape[1:])

    for item in obj:
        if depth > 1:
            if not is_sequence_of(item, types, depth=depth - 1,
                                  shape=next_shape):
                return False
        elif types is not None and not isinstance(item, types):
            return False
    return True


def is_function(f: Callable[..., Any],
                arg_count: int, coroutine: bool = False) -> bool:
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


# could use numpy.arange here, but
# a) we don't want to require that as a dep so low level
# b) I'd like to be more flexible with the sign of step
def permissive_range(start: float, stop: float, step: SupportsAbs[float]
                     ) -> List[float]:
    """
    Returns a range (as a list of values) with floating point steps.
    Always starts at start and moves toward stop, regardless of the
    sign of step.

    Args:
        start: The starting value of the range.
        stop: The end value of the range.
        step: Spacing between the values.
    """
    signed_step = abs(step) * (1 if stop > start else -1)
    # take off a tiny bit for rounding errors
    step_count = math.ceil((stop - start) / signed_step - 1e-10)
    return [start + i * signed_step for i in range(step_count)]


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


class DelegateAttributes:
    """
    Mixin class to create attributes of this object by
    delegating them to one or more dictionaries and/or objects.

    Also fixes ``__dir__`` so the delegated attributes will show up
    in ``dir()`` and ``autocomplete``.

    Attribute resolution order:
        1. Real attributes of this object.
        2. Keys of each dictionary in ``delegate_attr_dicts`` (in order).
        3. Attributes of each object in ``delegate_attr_objects`` (in order).
    """
    delegate_attr_dicts: List[str] = []
    """
    A list of names (strings) of dictionaries
    which are (or will be) attributes of ``self``, whose keys should
    be treated as attributes of ``self``.
    """
    delegate_attr_objects: List[str] = []
    """
    A list of names (strings) of objects
    which are (or will be) attributes of ``self``, whose attributes
    should be passed through to ``self``.
    """
    omit_delegate_attrs: List[str] = []
    """
    A list of attribute names (strings)
    to *not* delegate to any other dictionary or object.
    """

    def __getattr__(self, key: str) -> Any:
        if key in self.omit_delegate_attrs:
            raise AttributeError("'{}' does not delegate attribute {}".format(
                self.__class__.__name__, key))

        for name in self.delegate_attr_dicts:
            if key == name:
                # needed to prevent infinite loops!
                raise AttributeError(
                    "dict '{}' has not been created in object '{}'".format(
                        key, self.__class__.__name__))
            try:
                d = getattr(self, name, None)
                if d is not None:
                    return d[key]
            except KeyError:
                pass

        for name in self.delegate_attr_objects:
            if key == name:
                raise AttributeError(
                    "object '{}' has not been created in object '{}'".format(
                        key, self.__class__.__name__))
            try:
                obj = getattr(self, name, None)
                if obj is not None:
                    return getattr(obj, key)
            except AttributeError:
                pass

        raise AttributeError(
            "'{}' object and its delegates have no attribute '{}'".format(
                self.__class__.__name__, key))

    def __dir__(self) -> List[str]:
        names = list(super().__dir__())
        for name in self.delegate_attr_dicts:
            d = getattr(self, name, None)
            if d is not None:
                names += [k for k in d.keys()
                          if k not in self.omit_delegate_attrs]

        for name in self.delegate_attr_objects:
            obj = getattr(self, name, None)
            if obj is not None:
                names += [k for k in dir(obj)
                          if k not in self.omit_delegate_attrs]

        return sorted(set(names))


def strip_attrs(obj: object, whitelist: Sequence[str] = ()) -> None:
    """
    Irreversibly remove all direct instance attributes of object, to help with
    disposal, breaking circular references.

    Args:
        obj: Object to be stripped.
        whitelist: List of names that are not stripped from the object.
    """
    try:
        lst = set(list(obj.__dict__.keys())) - set(whitelist)
        for key in lst:
            try:
                del obj.__dict__[key]
            # TODO (giulioungaretti) fix bare-except
            except:
                pass
        # TODO (giulioungaretti) fix bare-except
    except:
        pass


def compare_dictionaries(dict_1: Dict[Hashable, Any],
                         dict_2: Dict[Hashable, Any],
                         dict_1_name: Optional[str] = 'd1',
                         dict_2_name: Optional[str] = 'd2',
                         path: str = "") -> Tuple[bool, str]:
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
    err = ''
    key_err = ''
    value_err = ''
    old_path = path
    for k in dict_1.keys():
        path = old_path + "[%s]" % k
        if k not in dict_2.keys():
            key_err += "Key {}{} not in {}\n".format(
                dict_1_name, path, dict_2_name)
        else:
            if isinstance(dict_1[k], dict) and isinstance(dict_2[k], dict):
                err += compare_dictionaries(dict_1[k], dict_2[k],
                                            dict_1_name, dict_2_name, path)[1]
            else:
                match = (dict_1[k] == dict_2[k])

                # if values are equal-length numpy arrays, the result of
                # "==" is a bool array, so we need to 'all' it.
                # In any other case "==" returns a bool
                # TODO(alexcjohnson): actually, if *one* is a numpy array
                # and the other is another sequence with the same entries,
                # this will compare them as equal. Do we want this, or should
                # we require exact type match?
                if hasattr(match, 'all'):
                    match = match.all()

                if not match:
                    value_err += (
                        'Value of "{}{}" ("{}", type"{}") not same as\n'
                        '  "{}{}" ("{}", type"{}")\n\n').format(
                        dict_1_name, path, dict_1[k], type(dict_1[k]),
                        dict_2_name, path, dict_2[k], type(dict_2[k]))

    for k in dict_2.keys():
        path = old_path + f"[{k}]"
        if k not in dict_1.keys():
            key_err += "Key {}{} not in {}\n".format(
                dict_2_name, path, dict_1_name)

    dict_differences = key_err + value_err + err
    if len(dict_differences) == 0:
        dicts_equal = True
    else:
        dicts_equal = False
    return dicts_equal, dict_differences


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
    We want to have a partial function which will allow us access the docstring
    through the python built-in help function. This is particularly important
    for client-facing driver methods, whose arguments might not be obvious.

    Consider the follow example why this is needed:

    >>> from functools import partial
    >>> def f():
    >>> ... pass
    >>> g = partial(f)
    >>> g.__doc__ = "bla"
    >>> help(g) # this will print an unhelpful message

    Args:
        func: A function that its docstring will be accessed.
        docstring: The docstring of the corresponding function.
    """
    ex = partial(func, **kwargs)

    def inner(**inner_kwargs: Any) -> None:
        ex(**inner_kwargs)

    inner.__doc__ = docstring

    return inner


def create_on_off_val_mapping(on_val: Any = True, off_val: Any = False
                              ) -> Dict[Union[str, bool], Any]:
    """
    Returns a value mapping which maps inputs which reasonably mean "on"/"off"
    to the specified ``on_val``/``off_val`` which are to be sent to the
    instrument. This value mapping is such that, when inverted,
    ``on_val``/``off_val`` are mapped to boolean ``True``/``False``.
    """
    # Here are the lists of inputs which "reasonably" mean the same as
    # "on"/"off" (note that True/False values will be added below, and they
    # will always be added)
    ons_: Tuple[Union[str, bool], ...] = ('On',  'ON',  'on',  '1')
    offs_: Tuple[Union[str, bool], ...] = ('Off', 'OFF', 'off', '0')

    # The True/False values are added at the end of on/off inputs,
    # so that after inversion True/False will be the only remaining
    # keys in the inverted value mapping dictionary.
    # NOTE that using 1/0 integer values will also work implicitly
    # due to `hash(True) == hash(1)`/`hash(False) == hash(0)`,
    # hence there is no need for adding 1/0 values explicitly to
    # the list of `ons` and `offs` values.
    ons = ons_ + (True,)
    offs = offs_ + (False,)

    return OrderedDict([(on, on_val) for on in ons]
                       + [(off, off_val) for off in offs])


def abstractmethod(funcobj: Callable[..., Any]) -> Callable[..., Any]:
    """
    A decorator indicating abstract methods.

    This is heavily inspired by the decorator of the same name in
    the ABC standard library. But we make our own version because
    we actually want to allow the class with the abstract method to be
    instantiated and we will use this property to detect if the
    method is abstract and should be overwritten.
    """
    funcobj.__qcodes_is_abstract_method__ = True  # type: ignore[attr-defined]
    return funcobj


def _ruamel_importer() -> type:
    try:
        from ruamel_yaml import YAML
    except ImportError:
        try:
            from ruamel.yaml import YAML
        except ImportError:
            raise ImportError('No ruamel module found. Please install '
                              'either ruamel.yaml or ruamel_yaml.')
    return YAML


# YAML module to be imported. Resovles naming issues of YAML from pypi and
# anaconda
YAML = _ruamel_importer()


def get_qcodes_path(*subfolder: str) -> str:
    """
    Return full file path of the QCoDeS module. Additional arguments will be
    appended as subfolder.

    """
    import qcodes
    path = os.sep.join(qcodes.__file__.split(os.sep)[:-1])
    return os.path.join(path, *subfolder) + os.sep


def get_qcodes_user_path(*file_parts: str) -> str:
    """
    Get ``~/.qcodes`` path or if defined the path defined in the
    ``QCODES_USER_PATH`` environment variable.

    Returns:
        path to the user qcodes directory

    """
    path = os.environ.get(QCODES_USER_PATH_ENV,
                          os.path.join(Path.home(), '.qcodes'))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return os.path.join(path, *file_parts)


def checked_getattr(
    instance: Any,
    attribute: str,
    expected_type: Union[type, Tuple[type, ...]]
) -> Any:
    """
    Like ``getattr`` but raises type error if not of expected type.
    """
    attr: Any = getattr(instance, attribute)
    if not isinstance(attr, expected_type):
        raise TypeError()
    return attr
