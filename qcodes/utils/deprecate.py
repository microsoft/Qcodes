import warnings
import types
from contextlib import contextmanager
from typing import Optional, Callable, Any, cast, Iterator, List

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
) -> Callable[..., Any]:
    """
    A utility function to decorate deprecated functions and classes.

    Args:
        reason: The reason of deprecation.
        alternative: The alternative function or class to put in use instead of
            the deprecated one.

    """

    @wrapt.decorator  # type: ignore[misc]
    def decorate_callable(func: Callable[..., Any],
                          instance: object, args: Any, kwargs: Any) -> Any:
        t, n = (('class', instance.__class__.__name__)
                if func.__name__ == '__init__'
                else ('function', func.__name__))
        issue_deprecation_warning(f'{t} <{n}>', reason, alternative)
        return func(*args, **kwargs)

    def actual_decorator(obj: Any) -> Any:
        if isinstance(obj, (types.FunctionType, types.MethodType)):
            func = cast(Callable[..., Any], obj)
            # pylint: disable=no-value-for-parameter
            return decorate_callable(func)
            # pylint: enable=no-value-for-parameter
        else:
            # this would need to be recursive
            for m_name in dir(obj):
                m = getattr(obj, m_name)
                if isinstance(m, (types.FunctionType, types.MethodType)):
                    # skip static methods, since they are not wrapped correctly
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
def _catch_deprecation_warnings() -> Iterator[List[warnings.WarningMessage]]:
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("ignore")
        warnings.filterwarnings("always", category=QCoDeSDeprecationWarning)
        yield ws


@contextmanager
def assert_not_deprecated() -> Iterator[None]:
    with _catch_deprecation_warnings() as ws:
        yield
    assert len(ws) == 0


@contextmanager
def assert_deprecated(message: str) -> Iterator[None]:
    with _catch_deprecation_warnings() as ws:
        yield
    assert len(ws) == 1
    recorded_message = ws[0].message
    assert isinstance(recorded_message, Warning)
    assert recorded_message.args[0] == message
