import signal
import logging
import threading

log = logging.getLogger(__name__)


class DelayedKeyboardInterrupt:
    """
    A context manager to wrap a piece of code to ensure that a
    KeyboardInterrupt is not triggered by a SIGINT during the execution of
    this context. A second SIGINT will trigger the KeyboardInterrupt
    immediately.

    Inspired by https://stackoverflow.com/questions/842557/how-to-prevent-a-block-of-code-from-being-interrupted-by-keyboardinterrupt-in-py
    """
    signal_received = None
    old_handler = None

    def __enter__(self) -> None:
        is_main_thread = threading.current_thread() is threading.main_thread()
        is_default_sig_handler = (signal.getsignal(signal.SIGINT)
                                  is signal.default_int_handler)
        if is_default_sig_handler and is_main_thread:
            self.old_handler = signal.signal(signal.SIGINT, self.handler)
        elif is_default_sig_handler:
            log.debug("Not on main thread cannot intercept interrupts")

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        print("Received SIGINT, Will interrupt at first suitable time. "
              "Send second SIGINT to interrupt immediately.")
        # now that we have gotten one SIGINT make the signal
        # trigger a keyboard interrupt normally
        signal.signal(signal.SIGINT, self.forceful_handler)
        log.info('SIGINT received. Delaying KeyboardInterrupt.')

    @staticmethod
    def forceful_handler(sig, frame):
        print("Second SIGINT received. Triggering "
              "KeyboardInterrupt immediately.")
        log.info('Second SIGINT received. Triggering '
                 'KeyboardInterrupt immediately.')
        signal.default_int_handler(sig, frame)

    def __exit__(self, exception_type, value, traceback):
        if self.old_handler is not None:
            signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received is not None:
            self.old_handler(*self.signal_received)
