import logging
import threading
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_LOG = logging.getLogger(__name__)


class RespondingThread[T](threading.Thread):
    """
    Thread subclass for parallelizing execution. Behaves like a
    regular thread but returns a value from target, and propagates
    exceptions back to the main thread when this value is collected.

    The `output` method joins the thread, then checks for errors and
    returns the output value.

    so, if you have a function `f` where `f(1, 2, a=3) == 6`, then:

    >>> def f(x, y, a=0):
    ...     return x + y + a
    >>> thread = RespondingThread(target=f, args=(1, 2), kwargs={'a': 3})
    >>> thread.start()
    >>> out = thread.output()
    >>> out
    6
    """

    def __init__(
        self,
        target: "Callable[..., T]",
        args: "Sequence[Any]" = (),
        kwargs: dict[str, Any] | None = None,
        *args2: Any,
        **kwargs2: Any,
    ):
        if kwargs is None:
            kwargs = {}

        super().__init__(*args2, **kwargs2)

        self._target = target
        self._args = args
        self._kwargs = kwargs
        self._exception: Exception | None = None
        self._output: T | None = None

    def run(self) -> None:
        _LOG.debug(f"Executing {self._target} on thread: {threading.get_ident()}")
        try:
            self._output = self._target(*self._args, **self._kwargs)
        except Exception as e:
            self._exception = e

    def output(self, timeout: float | None = None) -> T | None:
        self.join(timeout=timeout)

        if self._exception:
            e = self._exception
            self._exception = None
            raise e

        return self._output


def thread_map[T](
    callables: "Sequence[Callable[..., T]]",
    args: Optional["Sequence[Sequence[Any]]"] = None,
    kwargs: Optional["Sequence[dict[str, Any]]"] = None,
) -> list[T | None]:
    """
    Evaluate a sequence of callables in separate threads, returning
    a list of their return values.

    Args:
        callables: A sequence of callables.
        args (Optional): A sequence of sequences containing the positional
            arguments for each callable.
        kwargs (Optional): A sequence of dicts containing the keyword arguments
            for each callable.

    """
    if args is None:
        args = ((),) * len(callables)
    if kwargs is None:
        empty_dict: dict[str, Any] = {}
        kwargs = (empty_dict,) * len(callables)
    threads = [
        RespondingThread(target=c, args=a, kwargs=k)
        for c, a, k in zip(callables, args, kwargs)
    ]

    for t in threads:
        t.start()

    return [t.output() for t in threads]


if not TYPE_CHECKING:
    from typing import TypeVar

    from qcodes.utils.deprecate import _make_deprecated_typevars_getattr

    __getattr__ = _make_deprecated_typevars_getattr(
        __name__,
        {
            "T": TypeVar("T"),
        },
    )
