"""multiprocessing helper functions."""

import multiprocessing as mp
import time
import warnings

MP_ERR = 'context has already been set'


def set_mp_method(method, force=False):
    """
    An idempotent wrapper for multiprocessing.set_start_method.

    The most important use of this is to force Windows behavior
    on a Mac or Linux: set_mp_method('spawn')
    args are the same:

    method: one of:
        'fork' (default on unix/mac)
        'spawn' (default, and only option, on windows)
        'forkserver'
    force: allow changing context? default False
        in the original function, even calling the function again
        with the *same* method raises an error, but here we only
        raise the error if you *don't* force *and* the context changes
    """
    warnings.warn("Multiprocessing is in beta, use at own risk", UserWarning)
    try:
        mp.set_start_method(method, force=force)
    except RuntimeError as err:
        if err.args != (MP_ERR, ):
            raise

    mp_method = mp.get_start_method()
    if mp_method != method:
        raise RuntimeError(
            'unexpected multiprocessing method '
            '\'{}\' when trying to set \'{}\''.format(mp_method, method))


def kill_queue(queue):
    """Tear down a multiprocessing.Queue to help garbage collection."""
    try:
        queue.close()
        queue.join_thread()
    except:
        pass


def kill_processes():
    """Kill all running child processes."""
    # TODO: Instrument processes don't appropriately stop in all tests...
    for process in mp.active_children():
        try:
            process.terminate()
        except:
            pass

    if mp.active_children():
        time.sleep(0.2)
