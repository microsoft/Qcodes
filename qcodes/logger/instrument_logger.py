from contextlib import contextmanager
import logging
from typing import Optional, Sequence, Union, TYPE_CHECKING
import collections.abc
from .logger import get_console_handler, LevelType, handler_level
if TYPE_CHECKING:
    import qcodes.instrument.InstrumentBase as InstrumentBase  # noqa: F401 pylint: disable=unused-import


class InstrumentLoggerAdapter(logging.LoggerAdapter):
    """
    In the python logging module adapters are used to add context information
    to logging. The `LoggerAdapter` has the same methods as the `Logger` and
    can thus be used as such.

    Here it is used to add the instruments full name to the log records so that
    they can be filetered by the `InstrumentFilter` by the instrument instance.

    The context data gets stored in the `extra` dictionary as a property of the
    Adapter. It is filled by the `__init__` method:
    >>> LoggerAdapter(log, {'instrument': self.full_name})
    """
    def process(self, msg, kwargs):
        """
        returns the message and the kwargs for the handlers.
        """
        kwargs['extra'] = self.extra
        inst = self.extra['instrument']
        return f"[{inst.full_name}({type(inst).__name__})] {msg}", kwargs


class InstrumentFilter(logging.Filter):
    """
    Filter to filter out records that originate from the given instruments.
    Records created through the `InstrumentLoggerAdapter` have additional
    properties as specified in the `extra` dictionary which is a property of
    the adapter.

    Here the `instrument` property gets used to reject records that don't have
    originate from the list of instruments that has been passed to the
    `__init__`
    """
    def __init__(self, instruments):
        # This local import is necessary to avoid a circular import dependency.
        # The alternative is to merge this module with the instrument.base,
        # which is also not favorable.
        super().__init__()
        if not isinstance(instruments, collections.abc.Sequence):
            instrument_seq = (instruments,)
        else:
            instrument_seq = instruments
        self.instrument_set = set(instrument_seq)

    def filter(self, record):
        try:
            inst = record.instrument
        except AttributeError:
            return False
        return not self.instrument_set.isdisjoint(inst.ancestors)


def get_instrument_logger(instrument_instance: 'InstrumentBase',
                          logger_name: Optional[str]=None
                          ) -> InstrumentLoggerAdapter:
    """
    Return an `InstrumentLoggerAdapter` that can be used to log messages
    including `instrument_instance` as  an additional context.

    The `LoggerAdapter` object can be used as any logger.

    Args:
        instrument_instance: the instrument instance to be added to the context
            of the log record.
        logger_name: name of the logger to which the records will be passed.
            If `None`, defaults to the root logger.

    Returns:
        LoggerAdapter instance, that can be used for instrument specific
        logging.
    """
    logger_name = logger_name or ''
    return InstrumentLoggerAdapter(logging.getLogger(logger_name),
                                   {'instrument': instrument_instance})


@contextmanager
def filter_instrument(instrument: Union['InstrumentBase',
                                        Sequence['InstrumentBase']],
                      handler: Optional[
                          Union[logging.Handler,
                                Sequence[logging.Handler]]]=None,
                      level: Optional[LevelType]=None):
    """
    Context manager that adds a filter that only enables the log messages of
    the supplied instruments to pass.
    Example:
        >>> h1, h2 = logger.get_console_handler(), logger.get_file_handler()
        >>> with logger.filter_instruments((qdac, dmm2), handler=[h1, h1]):
        >>>     qdac.ch01(1)  # logged
        >>>     v = dmm2.v()  # not logged

    Args:
        level: level to set the handlers to
        handler: single or sequence of handlers which to change
    """
    if handler is None:
        handler = (get_console_handler(),)
    if not isinstance(handler, collections.abc.Sequence):
        handler = (handler,)

    instrument_filter = InstrumentFilter(instrument)
    for h in handler:
        h.addFilter(instrument_filter)
    try:
        if level is not None:
            with handler_level(level, handler):
                yield
        else:
            yield
    finally:
        for h in handler:
            h.removeFilter(instrument_filter)
