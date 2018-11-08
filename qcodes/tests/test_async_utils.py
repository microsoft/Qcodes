from typing import TypeVar, Awaitable
T = TypeVar('T')

import asyncio

from enum import Enum, auto

from qcodes.utils.async_utils import sync, cancelling

class TaskCompletionState(Enum):
    not_started = auto()
    complete = auto()
    cancelled = auto()

async def sleepy_identity(x : T) -> T:
    # Don't wait very long, since we want this in a
    # testing context.
    await asyncio.sleep(0.02)
    return x

class TaskWatcher(object):
    state : TaskCompletionState = TaskCompletionState.not_started

    def reset(self) -> None:
        self.state = TaskCompletionState.not_started

    async def __call__(self, task : Awaitable[T]) -> T:
        try:
            val = await task
        except asyncio.CancelledError:
            self.state = TaskCompletionState.cancelled
            raise

        self.state = TaskCompletionState.complete
        return val


def test_sync():
    watcher = TaskWatcher()
    assert 42 == sync(watcher(sleepy_identity(42)))
    assert watcher.state == TaskCompletionState.complete

def test_cancelling():
    # Strategy: set up a task that we know can never
    #           complete, and use TaskWatcher to assert
    #           that the task is in fact cancelled.
    async def godot():
        while True:
            await asyncio.sleep(30)
    async def cancelable_task():
        await godot()
    cancellation_watcher = TaskWatcher()

    async def case():
        future = asyncio.ensure_future(
            cancellation_watcher(cancelable_task())
        )
        with cancelling(future):
            return await sleepy_identity(42)

    x = sync(case())
    assert x == 42
    assert cancellation_watcher.state == TaskCompletionState.cancelled
