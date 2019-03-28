# Instead of async, since there are really a very small set of things
# we want to happen simultaneously within one process (namely getting
# several parameters in parallel), we can parallelize them with threads.
# That way the things we call need not be rewritten explicitly async.

import threading


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
    """
    Evaluate a sequence of callables in separate threads, returning
    a list of their return values.

    Args:
        callables: a sequence of callables
        args (Optional): a sequence of sequences containing the positional
            arguments for each callable
        kwargs (Optional): a sequence of dicts containing the keyword arguments
            for each callable

    """
    if args is None:
        args = ((),) * len(callables)
    if kwargs is None:
        kwargs = ({},) * len(callables)
    threads = [RespondingThread(target=c, args=a, kwargs=k)
               for c, a, k in zip(callables, args, kwargs)]

    for t in threads:
        t.start()

    return [t.output() for t in threads]
