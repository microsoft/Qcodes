"""Sinks consume events from the engine's publisher thread.

A sink is any callable that accepts a single :py:class:`Event`. Sinks may
optionally declare a ``critical`` attribute (default ``False``): critical
sinks abort the run on failure, non-critical sinks merely log.

The default storage sink is :py:class:`SqliteSink`. The simplest sink is
:py:class:`MemorySink`, which records events into a list — useful for
tests and for in-memory consumption.
"""

from qcodes.measure_v2.sinks.memory import MemorySink
from qcodes.measure_v2.sinks.protocol import DataSink, is_critical
from qcodes.measure_v2.sinks.sqlite import SqliteSink

__all__ = ["DataSink", "MemorySink", "SqliteSink", "is_critical"]
