import os
import sys
import imp
from traceback import format_exc

# Routine to reload modules during execution, for development only.
# This is finicky and SHOULD NOT be used in regular experiments,
# as they can cause non-intuitive errors later on. It is not included in
# the base qcodes import, nor tested; Use at your own risk.


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
