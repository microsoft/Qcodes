from functools import wraps
import warnings
from typing import Optional, Callable


class QCoDeSDeprecationWarning(RuntimeWarning):
    """Fix for `DeprecationWarning` being suppressed by default."""

    pass


def issue_deprecation_warning(
    what: str,
    reason: Optional[str] = None,
    alternative: Optional[str] = None
) -> None:
    msg = f'The {what} is deprecated'
    if reason is not None:
        msg += f', because {reason}'
    msg += '.'
    if alternative is not None:
        msg += f' Use \"{alternative}\" as an alternative.'

    warnings.warn(msg, QCoDeSDeprecationWarning, stacklevel=2)


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
    def actual_decorator(func: Callable) -> Callable:
        @wraps(func)
        def decorated_func(*args, **kwargs):
            t, n = (('class', args[0].__class__.__name__)
                    if func.__name__ == '__init__'
                    else ('function', func.__name__))
            issue_deprecation_warning(f'{t} <{n}>', reason, alternative)
            return func(*args, **kwargs)
        return decorated_func
    return actual_decorator
