import multiprocessing as mp
import sys
from datetime import datetime


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


class StreamQueue(object):
    '''
    Redirect child process stdout and stderr to a queue

    One StreamQueue should be created in the consumer process, and passed
    to each child process.

    In the child, we call StreamQueue.connect with a process name that will be
    unique and meaningful to the user

    The consumer then periodically calls StreamQueue.get() to read these
    messages

    inspired by http://stackoverflow.com/questions/23947281/
    '''
    def __init__(self, *args, **kwargs):
        self.queue = mp.Queue(*args, **kwargs)
        self._last_stream = None
        self._on_new_line = True

    def connect(self, process_name):
        sys.stdout = _SQWriter(self, process_name, sys.__stdout__)
        sys.stderr = _SQWriter(self, process_name + ' ERR', sys.__stderr__)

    def disconnect(self):
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    def get(self):
        out = ''
        while not self.queue.empty():
            msg_time, stream_name, msg = self.queue.get()
            timestr = msg_time.strftime('%H:%M:%S:%f')[:-3]
            line_head = '[{} {}] '.format(timestr, stream_name)

            if self._on_new_line:
                out += line_head
            elif stream_name != self._last_stream:
                out += '\n' + line_head

            out += msg[:-1].replace('\n', '\n' + line_head) + msg[-1]

            self._on_new_line = (msg[-1] == '\n')
            self._last_stream = stream_name

        return out


class _SQWriter(object):
    def __init__(self, stream_queue, stream_name, base_stream):
        self.queue = stream_queue.queue
        self.stream_name = stream_name
        self.base_stream = base_stream

    def write(self, msg):
        self.queue.put((datetime.now(), self.stream_name, msg))

    def flush(self):
        # TODO - do I need the base_stream flush? doesn't seem necessary
        # we need a flush method but I don't see why it can't be a noop.
        # if that's OK, we can take out base_stream entirely.
        pass
        # self.base_stream.flush()
