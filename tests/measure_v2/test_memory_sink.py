"""Tests for the sink protocol and ``MemorySink``."""

from __future__ import annotations

import time
from uuid import uuid4

import pytest

from qcodes.measure_v2 import (
    DataSink,
    Descriptor,
    MemorySink,
    RowEmitted,
    RunStarted,
    RunStopped,
    is_critical,
)
from qcodes.parameters import Parameter


@pytest.fixture
def descriptor() -> Descriptor:
    g = Parameter("g")
    i = Parameter("i")
    return Descriptor(setpoints=(g,), measured=(i,))


def _started(descriptor: Descriptor) -> RunStarted:
    return RunStarted(
        run_id=uuid4(),
        name="t",
        descriptor=descriptor,
        exp=None,
        write_period=None,
        started_at=time.time(),
    )


def _stopped(run_id) -> RunStopped:
    return RunStopped(
        run_id=run_id,
        reason="completed",
        error=None,
        started_at=time.time(),
        stopped_at=time.time(),
    )


def test_memory_sink_records_events_in_order(descriptor: Descriptor) -> None:
    sink = MemorySink()
    start = _started(descriptor)
    row = RowEmitted(run_id=start.run_id, snapshot={}, seq=0)
    stop = _stopped(start.run_id)

    sink(start)
    sink(row)
    sink(stop)

    assert sink.events == [start, row, stop]


def test_memory_sink_convenience_accessors(descriptor: Descriptor) -> None:
    sink = MemorySink()
    start = _started(descriptor)
    sink(start)
    sink(RowEmitted(run_id=start.run_id, snapshot={}, seq=0))
    sink(RowEmitted(run_id=start.run_id, snapshot={}, seq=1))
    sink(_stopped(start.run_id))

    assert len(sink.rows) == 2
    assert len(sink.starts) == 1
    assert len(sink.stops) == 1
    assert sink.starts[0] is start


def test_memory_sink_clear(descriptor: Descriptor) -> None:
    sink = MemorySink()
    sink(_started(descriptor))
    sink.clear()
    assert sink.events == []


def test_memory_sink_is_non_critical_by_default() -> None:
    sink = MemorySink()
    assert is_critical(sink) is False


def test_is_critical_returns_false_for_plain_function() -> None:
    """A plain callable used as a sink defaults to non-critical."""

    def sink(event):
        pass

    assert is_critical(sink) is False


def test_is_critical_reads_attribute() -> None:
    """An object can opt into critical via an attribute."""

    class _CriticalSink:
        critical = True

        def __call__(self, event):
            pass

    assert is_critical(_CriticalSink()) is True


def test_memory_sink_satisfies_datasink_protocol() -> None:
    """Runtime protocol check: MemorySink is a DataSink."""
    sink = MemorySink()
    assert isinstance(sink, DataSink)


def test_plain_callable_satisfies_datasink_protocol() -> None:
    """Functions also satisfy the protocol (it's just a callable)."""

    def sink(event):
        pass

    assert isinstance(sink, DataSink)
