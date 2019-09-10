import signal
import logging


class DelayedKeyboardInterrupt:
    """
    A context manager to wrap a piece of code to ensure that a keyboard interupt is not
    triggered during the execution of this code.

    Inspired by https://stackoverflow.com/questions/842557/how-to-prevent-a-block-of-code-from-being-interrupted-by-keyboardinterrupt-in-py
    """
    signal_received = None
    old_handler = None

    def __enter__(self) -> None:
        if signal.getsignal(signal.SIGINT) is signal.default_int_handler:
            self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        logging.debug('SIGINT received. Delaying KeyboardInterrupt.')

    def __exit__(self, exception_type, value, traceback):
        if self.old_handler is not None:
            signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received is not None:
            self.old_handler(*self.signal_received)
