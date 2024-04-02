import types
import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Optional, cast

import wrapt  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from collections.abc import Iterator


class QCoDeSDeprecationWarning(RuntimeWarning):
    """
    A DeprecationWarning used internally in QCoDeS. This
    fixes `DeprecationWarning` being suppressed by default.
    """


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
    stacklevel: int = 3,
) -> None:
    """
    Issue a `QCoDeSDeprecationWarning` with a consistently formatted message
    """
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

    Note that this is not recommended to be used for deprecation any new code.
    It is recommended to use typing_extensions.deprecated (which will be added
    to the std lib as warnings.deprecated in python 3.13)

    Args:
        reason: The reason of deprecation.
        alternative: The alternative function or class to put in use instead of
            the deprecated one.

    """

    @wrapt.decorator  # type: ignore[misc]
    def decorate_callable(
        func: Callable[..., Any], instance: object, args: Any, kwargs: Any
    ) -> Any:
        t, n = (
            ("class", instance.__class__.__name__)
            if func.__name__ == "__init__"
            else ("function", func.__name__)
        )
        issue_deprecation_warning(f"{t} <{n}>", reason, alternative, stacklevel=3)
        return func(*args, **kwargs)

    def actual_decorator(obj: Any) -> Any:
        if isinstance(obj, (types.FunctionType, types.MethodType)):
            func = cast(Callable[..., Any], obj)

            return decorate_callable(func)  # pyright: ignore[reportCallIssue]
        else:
            # this would need to be recursive
            for m_name in dir(obj):
                m = getattr(obj, m_name)
                if isinstance(m, (types.FunctionType, types.MethodType)):
                    # skip static methods, since they are not wrapped correctly
                    # by wrapt.
                    # if anyone reading this knows how the following line
                    # works please let me know.
                    # wrapt cannot wrap class methods in 3.11.0
                    # see https://github.com/python/cpython/issues/63272
                    if isinstance(
                        obj.__dict__.get(m_name, None), (staticmethod, classmethod)
                    ):
                        continue

                    setattr(
                        obj,
                        m_name,
                        decorate_callable(m),  # pyright: ignore[reportCallIssue]
                    )
            return obj

    return actual_decorator


@contextmanager
def _catch_deprecation_warnings() -> "Iterator[list[warnings.WarningMessage]]":
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("ignore")
        warnings.filterwarnings("always", category=QCoDeSDeprecationWarning)
        yield ws


@contextmanager
def assert_not_deprecated() -> "Iterator[None]":
    with _catch_deprecation_warnings() as ws:
        yield
    assert len(ws) == 0


@contextmanager
def assert_deprecated(message: str) -> "Iterator[None]":
    with _catch_deprecation_warnings() as ws:
        yield
    assert len(ws) == 1
    recorded_message = ws[0].message
    assert isinstance(recorded_message, Warning)
    assert recorded_message.args[0] == message
