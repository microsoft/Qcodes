from asyncio import iscoroutinefunction
from collections import Iterable
from datetime import datetime
import imp
from inspect import getargspec, ismethod
import logging
import math
import multiprocessing as mp
import sys
import os
from traceback import format_exc


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

    argspec = getargspec(f)
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


def set_mp_method(method, force=False):
    '''
    an idempotent wrapper for multiprocessing.set_start_method
    args are the same:

    method: one of:
        'fork' (default on unix/mac)
        'spawn' (default, and only option, on windows)
        'forkserver'
    force: allow changing context? default False
        in the original function, even calling the function again
        with the *same* method raises an error, but here we only
        raise the error if you *don't* force *and* the context changes
    '''
    try:
        # force windows multiprocessing behavior on mac
        mp.set_start_method(method)
    except RuntimeError as err:
        if err.args != ('context has already been set', ):
            raise

    mp_method = mp.get_start_method()
    if mp_method != method:
        raise RuntimeError(
            'unexpected multiprocessing method '
            '\'{}\' when trying to set \'{}\''.format(mp_method, method))


class PrintableProcess(mp.Process):
    '''
    controls repr printing of the process
    subclasses should provide a `name` attribute to go in repr()
    if subclass.name = 'DataServer',
    repr results in eg '<DataServer-1, started daemon>'
    otherwise would be '<DataServerProcess(DataServerProcess...)>'
    '''
    def __repr__(self):
        cname = self.__class__.__name__
        out = super().__repr__().replace(cname + '(' + cname, self.name)
        return out.replace(')>', '>')


def safe_getattr(obj, key, attr_dict):
    '''
    __getattr__ delegation to avoid infinite recursion

    obj: the instance being queried
    key: the attribute name (string)
    attr_dict: the name (string) of the dict within obj that
        holds the attributes being delegated
    '''
    # TODO: is there a way to automatically combine this with __dir__?
    # because we're always just saying:
    #    def __getattr__(self, key):
    #        return safe_getattr(self, key, 'some_dict')
    # but then we should add self.some_dict's keys to dir() as well.
    # or should we set all these attributes up front, instead of using
    # __getattr__?
    try:
        if key == attr_dict:
            # we got here looking for the dict itself, but it doesn't exist
            msg = "'{}' object has no attribute '{}'".format(
                obj.__class__.__name__, key)
            raise AttributeError(msg)

        return getattr(obj, attr_dict)[key]

    except KeyError:
        msg = "'{}' object has no attribute or {} item '{}'".format(
            obj.__class__.__name__, attr_dict, key)
        raise AttributeError(msg) from None


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
