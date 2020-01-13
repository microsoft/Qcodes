from typing import TypeVar, Awaitable, Generator, List
import asyncio
from contextlib import contextmanager

T = TypeVar('T')

def sync(task : Awaitable[T]) -> T:
    """
    Given an awaitable task, runs that task synchronously
    in the default event loop. This function cannot be
    called from a thread that is already running an event loop.
    As a result, an async function called using sync may not
    call sync at any point in its call stack.
    """
    loop : asyncio.AbstractEventLoop = asyncio.get_event_loop()
    if loop.is_running():
        # We can't run in the same event loop that's already
        # running, so spawn a new thread.
        raise RuntimeError(
            r"Cannot run synchronously when an event loop is already running. "
            r"Please make sure to use sync() only at the outermost level in your "
                 r"call stack. "
            r"If you are using IPython's %autoawait extension, please use await the "
                 r"async version of this function/method, or disable %autoawait."
        )
    else:
        return loop.run_until_complete(task)

@contextmanager
def cancelling(*tasks : asyncio.Future) -> Generator[None, None, None]:
    """
    Given a sequence of cancellable futures, ensures that all
    futures are cancelled at the end of the context manager.
    Any exceptions raised during cancellation are aggregated.
    """
    try:
        yield
    finally:
        exceptions : List[Exception] = []
        for task in tasks:
            try:
                task.cancel()
            except Exception as ex:
                exceptions.append(ex)
        if exceptions:
            if len(exceptions) == 1:
                raise exceptions[0]
            else:
                raise RuntimeError(
                    "Multiple exceptions occurred when cancelling tasks:\n" +
                    "\n".join(
                        f"- {ex}" for ex in exceptions
                    )
                )

