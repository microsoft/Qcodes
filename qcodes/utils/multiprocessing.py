import multiprocessing as mp
import sys


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
