import io
import json
import logging
import math
import numbers
import sys
import time

from collections import Iterator, Sequence, Mapping
from copy import deepcopy

import numpy as np

_tprint_times = {}


class NumpyJSONEncoder(json.JSONEncoder):
    """Return numpy types as standard types."""
    # http://stackoverflow.com/questions/27050108/convert-numpy-type-to-python
    # http://stackoverflow.com/questions/9452775/converting-numpy-dtypes-to-native-python-types/11389998#11389998
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif (isinstance(obj, numbers.Complex) and
              not isinstance(obj, numbers.Real)):
            return {
                '__dtype__': 'complex',
                're': float(obj.real),
                'im': float(obj.imag)
            }
        else:
            return super(NumpyJSONEncoder, self).default(obj)


def tprint(string, dt=1, tag='default'):
    """ Print progress of a loop every dt seconds """
    ptime = _tprint_times.get(tag, 0)
    if (time.time() - ptime) > dt:
        print(string)
        _tprint_times[tag] = time.time()


def in_notebook():
    """
    Check if inside a notebook.
    This could mean we are connected to a notebook, but this is not guaranteed.
    see: http://stackoverflow.com/questions/15411967
    Returns:
        bool: True if the code is running with a ipython or jypyter

    """
    return 'ipy' in repr(sys.stdout)


def is_sequence(obj):
    """
    Test if an object is a sequence.

    We do not consider strings or unordered collections like sets to be
    sequences, but we do accept iterators (such as generators)
    """
    return (isinstance(obj, (Iterator, Sequence)) and
            not isinstance(obj, (str, bytes, io.IOBase)))


def is_sequence_of(obj, types, depth=1):
    """
    Test if object is a sequence of entirely certain class(es).

    Args:
        obj (any): the object to test.
        types (class or tuple of classes): allowed type(s)
        depth (int, optional): level of nesting, ie if depth=2 we expect
            a sequence of sequences. Default 1.
    Returns:
        bool, True if every item in ``obj`` matches ``types``
    """
    if not is_sequence(obj):
        return False
    for item in obj:
        if depth > 1:
            if not is_sequence_of(item, types, depth=depth - 1):
                return False
        elif not isinstance(item, types):
            return False
    return True


def full_class(obj):
    """The full importable path to an object's class."""
    return type(obj).__module__ + '.' + type(obj).__name__


def named_repr(obj):
    """Enhance the standard repr() with the object's name attribute."""
    s = '<{}.{}: {} at {}>'.format(
        obj.__module__,
        type(obj).__name__,
        str(obj.name),
        id(obj))
    return s


def deep_update(dest, update):
    """
    Recursively update one JSON structure with another.

    Only dives into nested dicts; lists get replaced completely.
    If the original value is a dict and the new value is not, or vice versa,
    we also replace the value completely.
    """
    for k, v_update in update.items():
        v_dest = dest.get(k)
        if isinstance(v_update, Mapping) and isinstance(v_dest, Mapping):
            deep_update(v_dest, v_update)
        else:
            dest[k] = deepcopy(v_update)
    return dest


# could use numpy.arange here, but
# a) we don't want to require that as a dep so low level
# b) I'd like to be more flexible with the sign of step
def permissive_range(start, stop, step):
    """
    returns range (as a list of values) with floating point step

    inputs:
        start, stop, step

    always starts at start and moves toward stop,
    regardless of the sign of step
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
def make_sweep(start, stop, step=None, num=None):
    """
    Generate numbers over a specified interval.
    Requires `start` and `stop` and (`step` or `num`)
    The sign of `step` is not relevant.

    Args:
        start (Union[int, float]): The starting value of the sequence.
        stop (Union[int, float]): The end value of the sequence.
        step (Optional[Union[int, float]]):  Spacing between values.
        num (Optional[int]): Number of values to generate.

    Returns:
        numpy.linespace: numbers over a specified interval.

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
        num = steps_lo + 1

    return np.linspace(start, stop, num=num).tolist()


def wait_secs(finish_clock):
    """
    calculate the number of seconds until a given clock time
    The clock time should be the result of time.perf_counter()
    Does NOT wait for this time.
    """
    delay = finish_clock - time.perf_counter()
    if delay < 0:
        logging.warning('negative delay {:.6f} sec'.format(delay))
        return 0
    return delay


class LogCapture():

    """
    context manager to grab all log messages, optionally
    from a specific logger

    usage:

    with LogCapture() as logs:
        code_that_makes_logs(...)
    log_str = logs.value
    """

    def __init__(self, logger=logging.getLogger()):
        self.logger = logger

    def __enter__(self):
        self.log_capture = io.StringIO()
        self.string_handler = logging.StreamHandler(self.log_capture)
        self.string_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(self.string_handler)
        return self

    def __exit__(self, type, value, tb):
        self.logger.removeHandler(self.string_handler)
        self.value = self.log_capture.getvalue()
        self.log_capture.close()


def make_unique(s, existing):
    """
    make string s unique, able to be added to a sequence `existing` of
    existing names without duplication, by appending _<int> to it if needed
    """
    n = 1
    s_out = s
    existing = set(existing)

    while s_out in existing:
        n += 1
        s_out = '{}_{}'.format(s, n)

    return s_out


class DelegateAttributes:
    """
    Mixin class to create attributes of this object by
    delegating them to one or more dicts and/or objects

    Also fixes __dir__ so the delegated attributes will show up
    in dir() and autocomplete


    Attributes:
        delegate_attr_dicts (list): a list of names (strings) of dictionaries
            which are (or will be) attributes of self, whose keys should
            be treated as attributes of self
        delegate_attr_objects (list): a list of names (strings) of objects
            which are (or will be) attributes of self, whose attributes
            should be passed through to self
        omit_delegate_attrs (list): a list of attribute names (strings)
            to *not* delegate to any other dict or object

    any `None` entry is ignored

    attribute resolution order:
        1. real attributes of this object
        2. keys of each dict in delegate_attr_dicts (in order)
        3. attributes of each object in delegate_attr_objects (in order)
    """
    delegate_attr_dicts = []
    delegate_attr_objects = []
    omit_delegate_attrs = []

    def __getattr__(self, key):
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

    def __dir__(self):
        names = super().__dir__()
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


def strip_attrs(obj, whitelist=()):
    """
    Irreversibly remove all direct instance attributes of obj, to help with
    disposal, breaking circular references.

    Args:
        obj:  object to be stripped
        whitelist (list): list of names that are not stripped from the object
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


def compare_dictionaries(dict_1, dict_2,
                         dict_1_name='d1',
                         dict_2_name='d2', path=""):
    """
    Compare two dictionaries recursively to find non matching elements

    Args:
        dict_1: dictionary 1
        dict_2: dictionary 2
        dict_1_name: optional name used in the differences string
        dict_2_name: ''
    Returns:
        dicts_equal:      Boolean
        dict_differences: formatted string containing the differences

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
        path = old_path + "[{}]".format(k)
        if k not in dict_1.keys():
            key_err += "Key {}{} not in {}\n".format(
                dict_2_name, path, dict_1_name)

    dict_differences = key_err + value_err + err
    if len(dict_differences) == 0:
        dicts_equal = True
    else:
        dicts_equal = False
    return dicts_equal, dict_differences
