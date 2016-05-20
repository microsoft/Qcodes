from collections import Iterable, Mapping
from copy import deepcopy
import time
import logging
import math
import sys
import io
import multiprocessing as mp
import numpy as np


def in_notebook():
    '''
    Returns True if the code is running with a ipython or jypyter
    This could mean we are connected to a notebook, but this is not guaranteed.
    see: http://stackoverflow.com/questions/15411967
    '''
    return 'ipy' in repr(sys.stdout)


def is_sequence(obj):
    '''
    is an object a sequence? We do not consider strings to be sequences,
    but note that mappings (dicts) and unordered sequences (sets) ARE
    sequences by this definition.
    '''
    return isinstance(obj, Iterable) and not isinstance(obj, (str, bytes))


def full_class(obj):
    """The full importable path to an object's class."""
    return type(obj).__module__ + '.' + type(obj).__name__


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
    '''
    returns range (as a list of values) with floating point step

    inputs:
        start, stop, step

    always starts at start and moves toward stop,
    regardless of the sign of step
    '''
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
    '''
    Requires `start` and `stop` and (`step` or `num`)
    The sign of `step` is not relevant.

    returns: a numpy.linespace(start, stop, num)

    Examples:
        make_sweep(0, 10, num=5)
        > [0.0, 2.5, 5.0, 7.5, 10.0]
        make_sweep(5, 10, step=1)
        > [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        make_sweep(15, 10.5, step=1.5)
        >[15.0, 13.5, 12.0, 10.5]
    '''
    if step and num:
        raise AttributeError('Don\'t use `step` and `num` at the same time.')
    if (step is None) and (num is None):
        raise ValueError('If you really want to go from `start` to '
                         '`stop` in one step, specify `num = 1`.')
    if step is not None:
        num_lo = np.floor((stop-start)/step)
        num_hi = np.ceil((stop-start)/step)

        if num_lo != num_hi:
            raise ValueError('Could not find an integer number of steps for '
                             'the the given `start`, `stop`, and `step` '
                             'values. \nNumber of steps is {:d} or {:d}.'
                             .format(abs(int(num_lo))+1, abs(int(num_hi))+1))
        num = abs(num_lo)+1

    return np.linspace(start, stop, num=num)


def wait_secs(finish_clock):
    '''
    calculate the number of seconds until a given clock time
    The clock time should be the result of time.perf_counter()
    Does NOT wait for this time.
    '''
    delay = finish_clock - time.perf_counter()
    if delay < 0:
        logging.warning('negative delay {:.6f} sec'.format(delay))
        return 0
    return delay


class LogCapture():
    '''
    context manager to grab all log messages, optionally
    from a specific logger
    '''
    def __init__(self, logger=logging.getLogger()):
        self.logger = logger

    def __enter__(self):
        self.log_capture = io.StringIO()
        self.string_handler = logging.StreamHandler(self.log_capture)
        self.string_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(self.string_handler)
        return self.log_capture

    def __exit__(self, type, value, tb):
        self.logger.removeHandler(self.string_handler)


def make_unique(s, existing):
    '''
    make string s unique, able to be added to a sequence `existing` of
    existing names without duplication, by appending _<int> to it if needed
    '''
    n = 1
    s_out = s
    existing = set(existing)

    while s_out in existing:
        n += 1
        s_out = '{}_{}'.format(s, n)

    return s_out


class DelegateAttributes:
    '''
    Mixin class to create attributes of this object by
    delegating them to one or more dicts and/or objects

    Also fixes __dir__ so the delegated attributes will show up
    in dir() and autocomplete

    delegate_attr_dicts: a list of names (strings) of dictionaries which are
        (or will be) attributes of self, whose keys should be treated as
        attributes of self
    delegate_attr_objects: a list of names (strings) of objects which are
        (or will be) attributes of self, whose attributes should be passed
        through to self
    omit_delegate_attrs: a list of attribute names (strings) to *not* delegate
        to any other dict or object

    any `None` entry is ignored

    attribute resolution order:
        1. real attributes of this object
        2. keys of each dict in delegate_attr_dicts (in order)
        3. attributes of each object in delegate_attr_objects (in order)
    '''
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


def strip_attrs(obj):
    '''
    Irreversibly remove all direct instance attributes of obj, to help with
    disposal, breaking circular references.
    '''
    try:
        for key in list(obj.__dict__.keys()):
            try:
                del obj.__dict__[key]
            except:
                pass
    except:
        pass


def killprocesses():
    # TODO: Instrument processes don't appropriately stop in all tests...
    # this just kills everything that's running.
    for process in mp.active_children():
        try:
            process.terminate()
        except:
            pass

    time.sleep(0.5)
