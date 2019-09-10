from functools import wraps
import warnings
from typing import Optional, Callable


def deprecate(
        reason: Optional[str] = None,
        alternative: Optional[str] = None
) -> Callable:
    def actual_decorator(func: Callable) -> Callable:
        @wraps(func)
        def decorated_func(*args, **kwargs):
            t, n = (('class', args[0].__class__.__name__)
                    if func.__name__ == '__init__'
                    else ('function', func.__name__))
            msg = f'The {t} <{n}> is deprecated'
            if reason is not None:
                msg += f', because {reason}'
            else:
                msg += '.'
            if alternative is not None:
                msg += f' Use \"{alternative}\" as an alternative.'

            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        return decorated_func
    return actual_decorator
