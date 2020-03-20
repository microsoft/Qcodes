# Instead of async, since there are really a very small set of things
# we want to happen simultaneously within one process (namely getting
# several parameters in parallel), we can parallelize them with threads.
# That way the things we call need not be rewritten explicitly async.

import threading
import ctypes
import time
from collections import Iterable
import logging
from IPython.lib import backgroundjobs as bg
from typing import Union

logger = logging.getLogger(__name__)


class RespondingThread(threading.Thread):
    """
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
    """

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
    """
    Evaluate a sequence of callables in separate threads, returning
    a list of their return values.

    Args:
        callables: a sequence of callables
        args (optional): a sequence of sequences containing the positional
            arguments for each callable
        kwargs (optional): a sequence of dicts containing the keyword arguments
            for each callable

    """
    if args is None:
        args = ((),) * len(callables)
    if kwargs is None:
        kwargs = ({},) * len(callables)
    threads = [
        RespondingThread(target=c, args=a, kwargs=k)
        for c, a, k in zip(callables, args, kwargs)
    ]

    for t in threads:
        t.start()

    return [t.output() for t in threads]


class UpdaterThread(threading.Thread):
    def __init__(
        self, callables, interval, name=None, max_threads=None, auto_start=True
    ):
        super().__init__(name=name)
        self._is_paused = False

        if not isinstance(callables, Iterable):
            callables = [callables]
        self.callables = callables

        self.interval = interval
        if max_threads is not None:
            active_threads = sum(
                thread.getName() == name for thread in threading.enumerate()
            )
            if active_threads > max_threads:
                logger.warning(f"Found {active_threads} active updater threads")

        if auto_start:
            self.start()

    def run(self):
        while not self._is_stopped:
            time.sleep(self.interval)
            if not self._is_paused:
                for callable in self.callables:
                    callable()
        else:
            logger.warning("Updater thread stopped")

    def pause(self):
        self._is_paused = True

    def unpause(self):
        self._is_paused = False

    def halt(self):
        self._is_stopped = True


def raise_exception_in_thread(
    thread: threading.Thread, exception_type: BaseException = SystemExit
):
    """Raises an exception in a thread, usually forcing it to terminate.

    Note that this can fail if the thread is in a try/except statement, causing
    unintended consequences. This should therefore only be used as a last resort

    Args:
        thread: thread for which to raise an exception
        exception_type: Type of exception to raise

    Returns:
        None
    """
    assert thread.is_alive(), "thread must be started"
    for tid, tobj in threading._active.items():
        if tobj is thread:
            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                tid, ctypes.py_object(exception_type)
            )
            if res == 0:
                raise ValueError("nonexistent thread id")
            elif res > 1:
                # """if it returns a number greater than one, you're in trouble,
                # and you should call it again with exc=NULL to revert the effect"""
                ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, 0)
                raise SystemError("PyThreadState_SetAsyncExc failed")
            return


class KillableThread(threading.Thread):
    def terminate(self):
        # must raise the SystemExit type, instead of a SystemExit() instance
        # due to a bug in PyThreadState_SetAsyncExc
        raise_exception_in_thread(self, SystemExit)


default_job_name = "_measurement"
allow_duplicate_jobs = False
job_manager = bg.BackgroundJobManager()


def active_job() -> Union[bg.BackgroundJobExpr, None]:
    """Return active job.
    This job must have been created by `new_job` with kwarg ``active=True``, and
    must still be running

    Returns:
        active job if it exists, else None
    """
    try:
        return next(job for job in job_manager.running if job.is_active)
    except StopIteration:
        return None

def last_active_job() -> Union[bg.BackgroundJobExpr, None]:
    """Return last active job.
    This job must have been created by `new_job` with kwarg ``active=True``, but
    does not need to be running anymore.

    Returns:
        last active job if it ever existed, else None
    """
    try:
        jobs = list(job_manager.all.values())
        return next(job for job in reversed(jobs) if job.is_active)
    except StopIteration:
        return None


def new_job(
    function, name: str = None, active: bool = True, daemon=False, *args, **kwargs
) -> bg.BackgroundJobFunc:
    """Run a function in a separate thread
    The thread is an IPython job, which has additional features that extend the
    default threading.Thread functionality. For instance, IPython jobs are
    registered via a job manager (see above)

    Args:
        function: Function to be called
        name: Thread name. If not set, ``default_job_name`` is used (see above)
        active: Whether to make the job active. If True, the function
            `active_job` will return this job if it is still running.
            There can only be one active job at the same time.
        daemon: Whether the thread should be allowed to continue living after
            the kernel has been shut down
        *args: Optional args to be passed to the function
        **kwargs: Optional kwargs to be passed to the function

    Returns:
        Newly created job

    Note:
        The job attribute ``is_active`` is added to the job after creation.
        The method ``terminate`` is added to the job after creation. This method
        is not foolproof (see `raise_exception_in_thread`).
    """
    if not name:
        name = default_job_name

    # Ensure thread does not already exist if allow_duplicate_threads = False
    if not allow_duplicate_jobs:
        existing_thread_names = [thread.name for thread in threading.enumerate()]
        if name in existing_thread_names:
            raise RuntimeError(f"Thread '{name}' already exists. Exiting")

    # Run thread as an IPython job, registered in the job manager
    job = job_manager.new(function, *args, daemon=daemon, kw=kwargs)

    job.name = name  # Thread name can only be set after the job has been created

    # Optionally register current job as active job
    job.is_active = active

    # Add terminate method
    job.terminate = lambda: raise_exception_in_thread(job)

    return job
