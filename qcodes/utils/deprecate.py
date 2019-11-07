import warnings
import types
from contextlib import contextmanager
from functools import partial
from typing import Optional, Callable, Any, cast

import wrapt


class QCoDeSDeprecationWarning(RuntimeWarning):
    """Fix for `DeprecationWarning` being suppressed by default."""


def deprecation_message(
    what: str,
    reason: Optional[str] = None,
    alternative: Optional[str] = None
) -> str:
    msg = f'The {what} is deprecated'
    if reason is not None:
        msg += f', because {reason}'
    msg += '.'
    if alternative is not None:
        msg += f' Use \"{alternative}\" as an alternative.'
    return msg


def issue_deprecation_warning(
    what: str,
    reason: Optional[str] = None,
    alternative: Optional[str] = None,
    stacklevel: int = 2
) -> None:
    warnings.warn(
        deprecation_message(what, reason, alternative),
        QCoDeSDeprecationWarning,
        stacklevel=stacklevel)


def deprecate(
        reason: Optional[str] = None,
        alternative: Optional[str] = None
) -> Callable:
    """
    A utility function to decorate deprecated functions and classes.

    Args:
        reason: The reason of deprecation.
        alternative: The alternative function or class to put in use instead of
            the deprecated one.

    """

    @wrapt.decorator
    def decorate_callable(func, instance, args, kwargs):
        t, n = (('class', instance.__class__.__name__)
                if func.__name__ == '__init__'
                else ('function', func.__name__))
        issue_deprecation_warning(f'{t} <{n}>', reason, alternative)
        return func(*args, **kwargs)

    def actual_decorator(obj: Any) -> Any:
        if isinstance(obj, (types.FunctionType, types.MethodType)):
            func = cast(Callable, obj)
            # pylint: disable=no-value-for-parameter
            return decorate_callable(func)
            # pylint: enable=no-value-for-parameter
        else:
            # this would need to be recursive
            for m_name in dir(obj):
                m = getattr(obj, m_name)
                if isinstance(m, (types.FunctionType, types.MethodType)):
                    # skip static methods, since they are not wrapped corectly
                    # by wrapt.
                    # if anyone reading this knows how the following line
                    # works please let me know.
                    if isinstance(obj.__dict__.get(m_name, None), staticmethod):
                        continue
                    # pylint: disable=no-value-for-parameter
                    setattr(obj, m_name, decorate_callable(m))
                    # pylint: enable=no-value-for-parameter
            return obj

    return actual_decorator

@contextmanager
def _catch_deprecation_warnings():
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("ignore")
        warnings.filterwarnings("always", category=QCoDeSDeprecationWarning)
        yield ws


@contextmanager
def assert_not_deprecated():
    with _catch_deprecation_warnings() as ws:
        yield
    assert len(ws) == 0


@contextmanager
def assert_deprecated(message: str):
    with _catch_deprecation_warnings() as ws:
        yield
    assert len(ws) == 1
    assert ws[0].message.args[0] == message


deprecate_moved_to_qcd = partial(deprecate, reason="This driver has been moved "
                                                   "to Qcodes_contrib_drivers "
                                                   "and will be removed "
                                                   "from QCoDeS eventually.")
