import multiprocessing as mp
import sys
from datetime import datetime
import time

from .helpers import in_notebook


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
        # make sure the singleton StreamQueue exists
        # prior to launching a new process
        if queue_streams and in_notebook():
            self.stream_queue = get_stream_queue()
        else:
            self.stream_queue = None
        super().__init__(*args, name=name, daemon=daemon, **kwargs)

    def run(self):
        if self.stream_queue:
            self.stream_queue.connect(str(self.name))
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
    Do not instantiate this directly: use get_stream_queue so we only make one.

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
        self.last_read_ts = mp.Value('d', time.time())
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
            timestr, stream_name, msg = self.queue.get()
            line_head = '[{} {}] '.format(timestr, stream_name)

            if self._on_new_line:
                out += line_head
            elif stream_name != self._last_stream:
                out += '\n' + line_head

            out += msg[:-1].replace('\n', '\n' + line_head) + msg[-1]

            self._on_new_line = (msg[-1] == '\n')
            self._last_stream = stream_name

        self.last_read_ts.value = time.time()
        return out


class _SQWriter(object):
    MIN_READ_TIME = 3

    def __init__(self, stream_queue, stream_name):
        self.queue = stream_queue.queue
        self.last_read_ts = stream_queue.last_read_ts
        self.stream_name = stream_name

    def write(self, msg):
        try:
            if msg:
                msgtuple = (datetime.now().strftime('%H:%M:%S.%f')[:-3],
                            self.stream_name, msg)
                self.queue.put(msgtuple)

                queue_age = time.time() - self.last_read_ts.value
                if queue_age > self.MIN_READ_TIME and msg != '\n':
                    # long time since the queue was read? maybe nobody is
                    # watching it at all - send messages to the terminal too
                    # but they'll still be in the queue if someone DOES look.
                    termstr = '[{} {}] {}'.format(*msgtuple)
                    # we always want a new line this way (so I don't use
                    # end='' in the print) but we don't want an extra if the
                    # caller already included a newline.
                    if termstr[-1] == '\n':
                        termstr = termstr[:-1]
                    print(termstr, file=sys.__stdout__)
        except:
            # don't want to get an infinite loop if there's something wrong
            # with the queue - put the regular streams back before handling
            sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
            raise

    def flush(self):
        pass
