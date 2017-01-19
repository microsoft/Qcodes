"""Modifications to multiprocessing.Process common to all Qcodes processes."""

import multiprocessing as mp
from traceback import print_exc
import signal

from qcodes.utils.helpers import in_notebook

from .stream_queue import get_stream_queue


class QcodesProcess(mp.Process):

    """
    Modified multiprocessing.Process specialized to Qcodes needs.

    - Nicer repr
    - Automatic streaming of stdout and stderr to our StreamQueue singleton
      for reporting back to the main process
    - Ignore interrupt signals so that commands in the main process can be
      canceled without affecting server and background processes.
    """

    def __init__(self, *args, name='QcodesProcess', queue_streams=True,
                 daemon=True, **kwargs):
        """
        Construct the QcodesProcess, but like Process, do not start it.

        name: string to include in repr, and in the StreamQueue
            default 'QcodesProcess'
        queue_streams: should we connect stdout and stderr to the StreamQueue?
            default True
        daemon: should this process be treated as daemonic, so it gets
            terminated with the parent.
            default True, overriding the base inheritance
        any other args and kwargs are passed to multiprocessing.Process
        """
        # make sure the singleton StreamQueue exists
        # prior to launching a new process
        if queue_streams and in_notebook():
            self.stream_queue = get_stream_queue()
        else:
            self.stream_queue = None
        super().__init__(*args, name=name, daemon=daemon, **kwargs)

    def run(self):
        """Executed in the new process, and calls the target function."""
        # ignore interrupt signals, as they come from `KeyboardInterrupt`
        # which we want only to apply to the main process and not the
        # server and background processes (which can be halted in different
        # ways)
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        if self.stream_queue:
            self.stream_queue.connect(str(self.name))
        try:
            super().run()
        except:
            # if we let the system print the exception by itself, sometimes
            # it disconnects the stream partway through printing.
            print_exc()
        finally:
            if (self.stream_queue and
                    self.stream_queue.initial_streams is not None):
                self.stream_queue.disconnect()

    def __repr__(self):
        """Shorter and more helpful repr of our processes."""
        cname = self.__class__.__name__
        r = super().__repr__()
        r = r.replace(cname + '(', '').replace(')>', '>')
        return r.replace(', started daemon', '')
