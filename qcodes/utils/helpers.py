from asyncio import iscoroutinefunction
from collections import Iterable
from datetime import datetime
import imp
from inspect import getargspec, ismethod
import logging
import math
import sys
import os
from traceback import format_exc


def in_notebook():
    '''
    is this code in a process directly connected to a jupyter notebook?
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


def is_function(f, arg_count, coroutine=False):
    '''
    require a function with the specified number of positional arguments
    (and no kwargs) which either is or is not a coroutine
    type casting "functions" are allowed, but only in the 1-argument form
    '''
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
        argspec = getargspec(f)
    except TypeError:
        # some built-in functions/methods don't describe themselves to inspect
        # we already know it's a callable and coroutine is correct.
        return True

    if argspec.varargs:
        # we can't check the arg count if there's a *args parameter
        # so you're on your own at that point
        # unfortunately, the asyncio.coroutine decorator wraps with
        # *args and **kw so we can't count arguments with the old async
        # syntax, only the new syntax.
        return True

    # getargspec includes 'self' in the arg count, even though
    # it's not part of calling the function. So take it out.
    return len(argspec.args) - ismethod(f) == arg_count


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


def wait_secs(finish_datetime):
    '''
    calculate the number of seconds until a given datetime
    Does NOT wait for this time.
    '''
    delay = (finish_datetime - datetime.now()).total_seconds()
    if delay < 0:
        logging.warning('negative delay {} sec'.format(delay))
        return 0
    return delay


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


class DelegateAttributes(object):
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

    any `None` entry is ignored

    attribute resolution order:
        1. real attributes of this object
        2. keys of each dict in delegate_attr_dicts (in order)
        3. attributes of each object in delegate_attr_objects (in order)
    '''
    delegate_attr_dicts = []
    delegate_attr_objects = []

    def __getattr__(self, key):
        for name in self.delegate_attr_dicts:
            if key == name:
                # needed to prevent infinite loops!
                raise AttributeError(
                    "dict '{}' has not been created in object '{}'".format(
                        key, self.__class__.__name__))
            try:
                d = getattr(self, name)
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
                obj = getattr(self, name)
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
                names += list(d.keys())

        for name in self.delegate_attr_objects:
            obj = getattr(self, name, None)
            if obj is not None:
                names += dir(obj)

        return sorted(set(names))


# see http://stackoverflow.com/questions/22195382/
# how-to-check-if-a-module-library-package-is-part-of-the-python-standard-library
syspaths = [os.path.abspath(p) for p in sys.path]
stdlib = tuple(p for p in syspaths
               if p.startswith((sys.prefix, sys.base_prefix))
               and 'site-packages' not in p)
# a few things in site-packages we will consider part of the standard lib
# it causes problems if we reload some of these, others are just stable
# dependencies - this is mainly for reloading our own code.
# could even whitelist site-packages items to allow, rather than to ignore?
otherlib = ('jupyter', 'ipy', 'IPy', 'matplotlib', 'numpy', 'scipy', 'pyvisa',
            'traitlets', 'zmq', 'tornado', 'dateutil', 'six', 'pexpect')
otherpattern = tuple('site-packages/' + n for n in otherlib)


def reload_code(pattern=None, lib=False, site=False):
    '''
    reload all modules matching a given pattern
    or all (non-built-in) modules if pattern is omitted
    if lib is False (default), ignore the standard library and major packages
    if site is False (default), ignore everything in site-packages, only reload
    files in nonstandard paths
    '''
    reloaded_files = []

    for i in range(2):
        # sometimes we need to reload twice to propagate all links,
        # even though we reload the deepest modules first. Not sure if
        # twice is always sufficient, but we'll try it.
        for module in sys.modules.values():
            if (pattern is None or pattern in module.__name__):
                reload_recurse(module, reloaded_files, lib, site)

    return reloaded_files


def is_good_module(module, lib=False, site=False):
    '''
    is an object (module) a module we can reload?
    if lib is False (default), ignore the standard library and major packages
    '''
    # take out non-modules and underscore modules
    name = getattr(module, '__name__', '_')
    if name[0] == '_' or not isinstance(module, type(sys)):
        return False

    # take out modules we can't find and built-ins
    if name in sys.builtin_module_names or not hasattr(module, '__file__'):
        return False

    path = os.path.abspath(module.__file__)

    if 'site-packages' in path and not site:
        return False

    if not lib:
        if path.startswith(stdlib) and 'site-packages' not in path:
            return False

        for pattern in otherpattern:
            if pattern in path:
                return False

    return True


def reload_recurse(module, reloaded_files, lib, site):
    '''
    recursively search module for its own dependencies to reload,
    ignoring those already in reloaded_files
    if lib is False (default), ignore the standard library and major packages
    '''
    if (not is_good_module(module, lib, site) or
            module.__file__ in reloaded_files):
        return

    reloaded_files.append(module.__file__)

    try:
        for name in dir(module):
            module2 = getattr(module, name)
            reload_recurse(module2, reloaded_files, lib, site)
        imp.reload(module)

    except:
        print('error reloading "{}"'.format(getattr(module, '__name__', '?')))
        print(format_exc())
