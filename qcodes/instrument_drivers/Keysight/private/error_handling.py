from typing import Tuple

from qcodes.instrument.base import AbstractInstrument


class KeysightErrorQueueMixin(AbstractInstrument):
    """
    Mixin class for visa instruments that happen to implement an error queue.

    It is meant to work ONLY with drivers for Keysight
    instruments (which inherit from VisaInstrument class).
    """

    def error(self) -> Tuple[int, str]:
        """
        Return the first error message in the queue. It also clears it from
        the error queue.

        Up to 20 errors can be stored in the instrument's error queue.
        Error retrieval is first-in-first-out (FIFO).

        If more than 20 errors have occurred, the most recent error stored
        in the queue is replaced with -350,"Queue overflow". No additional
        errors are stored until you remove errors from the queue. If no
        errors have occurred when you read the error queue, the instrument
        responds with +0,"No error".

        Returns:
            The error code and the error message.
        """
        rawmssg = self.ask('SYSTem:ERRor?')
        code = int(rawmssg.split(',')[0])
        mssg = rawmssg.split(',')[1].strip().replace('"', '')

        return code, mssg

    def flush_error_queue(self, verbose: bool=True) -> None:
        """
        Clear the instrument error queue, and prints it.

        Args:
            verbose: If true, the error messages are printed.
                Default: True.
        """

        self.log.debug('Flushing error queue...')

        err_code, err_message = self.error()
        self.log.debug('    {}, {}'.format(err_code, err_message))
        if verbose:
            print(err_code, err_message)

        while err_code != 0:
            err_code, err_message = self.error()
            self.log.debug('    {}, {}'.format(err_code, err_message))
            if verbose:
                print(err_code, err_message)

        self.log.debug('...flushing complete')
