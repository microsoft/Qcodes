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


class QcodesProcess(mp.Process):
    '''
    modified multiprocessing.Process for nicer printing and automatic
    streaming of stdout and stderr to our StreamQueue singleton

    name: string to include in repr, and in the StreamQueue
        default 'QcodesProcess'
    queue_streams: should we connect stdout and stderr to the StreamQueue?
        default True
    daemon: should this process be treated as daemonic, so it gets terminated
        with the parent.
        default True, overriding the base inheritance
    any other args and kwargs are passed to multiprocessing.Process
    '''
    def __init__(self, *args, name='QcodesProcess', queue_streams=True,
                 daemon=True, **kwargs):
        # make sure the singleton exists prior to launching a new process
        self.queue_streams = queue_streams
        if queue_streams:
            get_stream_queue()
        super().__init__(*args, name=name, daemon=daemon, **kwargs)

    def run(self):
        if self.queue_streams:
            get_stream_queue().connect(str(self.name))
        super().run()

    def __repr__(self):
        cname = self.__class__.__name__
        return super().__repr__().replace(cname + '(', '').replace(')>', '>')


def get_stream_queue():
    '''
    convenience function to get a singleton StreamQueue
    note that this must be called from the main process before starting any
    subprocesses that will use it, otherwise the subprocess will create its
    own StreamQueue that no other processes know about
    '''
    if StreamQueue.instance is None:
        StreamQueue.instance = StreamQueue()
    return StreamQueue.instance


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
    instance = None

    def __init__(self, *args, **kwargs):
        self.queue = mp.Queue(*args, **kwargs)
        self._last_stream = None
        self._on_new_line = True

    def connect(self, process_name):
        sys.stdout = _SQWriter(self, process_name)
        sys.stderr = _SQWriter(self, process_name + ' ERR')

    def disconnect(self):
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    def get(self):
        out = ''
        while not self.queue.empty():
            msg_time, stream_name, msg = self.queue.get()
            timestr = msg_time.strftime('%H:%M:%S.%f')[:-3]
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
    def __init__(self, stream_queue, stream_name):
        self.queue = stream_queue.queue
        self.stream_name = stream_name

    def write(self, msg):
        if msg:
            self.queue.put((datetime.now(), self.stream_name, msg))

    def flush(self):
        pass
