"""StreamQueue: collect subprocess stdout/stderr to a single queue."""

import multiprocessing as mp
import sys
import time

from datetime import datetime

from .helpers import kill_queue


def get_stream_queue():
    """
    Convenience function to get a singleton StreamQueue.

    note that this must be called from the main process before starting any
    subprocesses that will use it, otherwise the subprocess will create its
    own StreamQueue that no other processes know about
    """
    if StreamQueue.instance is None:
        StreamQueue.instance = StreamQueue()
    return StreamQueue.instance


class StreamQueue:

    """
    Manages redirection of child process output for the main process to view.

    Do not instantiate this directly: use get_stream_queue so we only make one.
    One StreamQueue should be created in the consumer process, and passed
    to each child process. In the child, we call StreamQueue.connect with a
    process name that will be unique and meaningful to the user. The consumer
    then periodically calls StreamQueue.get() to read these messages.

    inspired by http://stackoverflow.com/questions/23947281/
    """

    instance = None

    def __init__(self, *args, **kwargs):
        """Create a StreamQueue, passing all args & kwargs to Queue."""
        self.queue = mp.Queue(*args, **kwargs)
        self.last_read_ts = mp.Value('d', time.time())
        self._last_stream = None
        self._on_new_line = True
        self.lock = mp.RLock()
        self.initial_streams = None

    def connect(self, process_name):
        """
        Connect a child process to the StreamQueue.

        After this, stdout and stderr go to a queue rather than being
        printed to a console.

        process_name: a short string that will clearly identify this process
            to the user.
        """
        if self.initial_streams is not None:
            raise RuntimeError('StreamQueue is already connected')

        self.initial_streams = (sys.stdout, sys.stderr)

        sys.stdout = _SQWriter(self, process_name)
        sys.stderr = _SQWriter(self, process_name + ' ERR')

    def disconnect(self):
        """Disconnect a child from the queues and revert stdout & stderr."""
        if self.initial_streams is None:
            raise RuntimeError('StreamQueue is not connected')
        sys.stdout, sys.stderr = self.initial_streams
        self.initial_streams = None

    def get(self):
        """Read new messages from the queue and format them for printing."""
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

    def __del__(self):
        """Tear down the StreamQueue either on the main or a child process."""
        try:
            self.disconnect()
        except:
            pass

        if hasattr(type(self), 'instance'):
            # so nobody else tries to use this dismantled stream queue later
            type(self).instance = None

        if hasattr(self, 'queue'):
            kill_queue(self.queue)
            del self.queue
        if hasattr(self, 'lock'):
            del self.lock


class _SQWriter:
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
                    try:
                        print(termstr, file=sys.__stdout__)
                    except ValueError:  # pragma: no cover
                        # ValueError: underlying buffer has been detached
                        # this may just occur in testing on Windows, not sure.
                        pass
        except:
            # don't want to get an infinite loop if there's something wrong
            # with the queue - put the regular streams back before handling
            sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
            raise

    def flush(self):
        pass
