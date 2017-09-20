# Instead of async, since there are really a very small set of things
# we want to happen simultaneously within one process (namely getting
# several parameters in parallel), we can parallelize them with threads.
# That way the things we call need not be rewritten explicitly async.

import threading
import ctypes
import time
from collections import Iterable
import logging

logger = logging.getLogger(__name__)


class RespondingThread(threading.Thread):
    '''
    a Thread subclass for parallelizing execution. Behaves like a
    regular thread but returns a value from target, and propagates
    exceptions back to the main thread when this value is collected.

    the `output` method joins the thread, then checks for errors and
    returns the output value.

    so, if you have a function `f` where `f(1, 2, a=3) == 4`, then:

    thread = RespondingThread(target=f, args=(1, 2), kwargs={'a': 3})
    thread.start()
    # do other things while this is running
    out = thread.output()  # out is 4
    '''
    def __init__(self, target=None, args=(), kwargs={}, *args2, **kwargs2):
        super().__init__(*args2, **kwargs2)

        self._target = target
        self._args = args
        self._kwargs = kwargs
        self._exception = None
        self._output = None

    def run(self):
        try:
            self._output = self._target(*self._args, **self._kwargs)
        except Exception as e:
            self._exception = e

    def output(self, timeout=None):
        self.join(timeout=timeout)

        if self._exception:
            e = self._exception
            self._exception = None
            raise e

        return self._output


def thread_map(callables, args=None, kwargs=None):
    '''
    Evaluate a sequence of callables in separate threads, returning
    a list of their return values.

    Args:
        callables: a sequence of callables
        args (optional): a sequence of sequences containing the positional
            arguments for each callable
        kwargs (optional): a sequence of dicts containing the keyword arguments
            for each callable

    '''
    if args is None:
        args = ((),) * len(callables)
    if kwargs is None:
        kwargs = ({},) * len(callables)
    threads = [RespondingThread(target=c, args=a, kwargs=k)
               for c, a, k in zip(callables, args, kwargs)]

    for t in threads:
        t.start()

    return [t.output() for t in threads]


class UpdaterThread(threading.Thread):
    def __init__(self, callables, interval, name=None, max_threads=None,
                 auto_start=True):
        super().__init__(name=name)
        self._is_paused = False

        if not isinstance(callables, Iterable):
            callables = [callables]
        self.callables = callables

        self.interval = interval
        if max_threads is not None:
            active_threads = sum(thread.getName()==name
                                 for thread in threading.enumerate())
            if  active_threads > max_threads:
                logger.warning(f'Found {active_threads} active updater threads')

        if auto_start:
            time.sleep(interval)
            self.start()

    def run(self):
        while not self._is_stopped:
            if not self._is_paused:
                for callable in self.callables:
                    callable()
            time.sleep(self.interval)
        else:
            logger.warning('Updater thread stopped')

    def pause(self):
        self._is_paused = True

    def unpause(self):
        self._is_paused = False

    def halt(self):
        self._is_stopped = True


def _async_raise(tid, excobj):
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(excobj))
    if res == 0:
        raise ValueError("nonexistent thread id")
    elif res > 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, 0)
        raise SystemError("PyThreadState_SetAsyncExc failed")


class KillableThread(threading.Thread):
    def raise_exc(self, excobj):
        assert self.isAlive(), "thread must be started"
        for tid, tobj in threading._active.items():
            if tobj is self:
                _async_raise(tid, excobj)
                return

        # the thread was alive when we entered the loop, but was not found
        # in the dict, hence it must have been already terminated. should we raise
        # an exception here? silently ignore?

    def terminate(self):
        # must raise the SystemExit type, instead of a SystemExit() instance
        # due to a bug in PyThreadState_SetAsyncExc
        self.raise_exc(SystemExit)
