"""In-memory sink: records every event into a list.

Useful for testing plan-builders and the engine without a database, and
for users who want to consume the event stream programmatically (e.g.,
into a custom analysis pipeline) without going through the SQLite sink.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qcodes.measure_v2.events import RowEmitted, RunStarted, RunStopped

if TYPE_CHECKING:
    from qcodes.measure_v2.events import Event


class MemorySink:
    """A sink that records all events into a list.

    Non-critical: an exception in a downstream sink does not affect the
    in-memory record. The recorded list is the authoritative log of what
    the engine emitted.

    Thread-safety: this sink is invoked from the engine's publisher thread,
    sequentially per event. Reads of ``events`` from other threads after
    the run has completed are safe; concurrent reads during a running
    measurement may see a partial list (it's a plain Python list).
    """

    critical: bool = False

    def __init__(self) -> None:
        self.events: list[Event] = []

    def __call__(self, event: Event) -> None:
        self.events.append(event)

    # --- Convenience accessors used heavily by tests ---

    @property
    def rows(self) -> list[RowEmitted]:
        """All :py:class:`RowEmitted` events, in arrival order."""
        return [e for e in self.events if isinstance(e, RowEmitted)]

    @property
    def starts(self) -> list[RunStarted]:
        """All :py:class:`RunStarted` events, in arrival order."""
        return [e for e in self.events if isinstance(e, RunStarted)]

    @property
    def stops(self) -> list[RunStopped]:
        """All :py:class:`RunStopped` events, in arrival order."""
        return [e for e in self.events if isinstance(e, RunStopped)]

    def clear(self) -> None:
        """Drop all recorded events (e.g., between runs in a test)."""
        self.events.clear()
