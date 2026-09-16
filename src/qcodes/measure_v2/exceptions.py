"""Exceptions used by the measure_v2 plan/engine API.

Plan authors only need to know about :py:class:`CancelRequested`: catch it
only if you need to distinguish cancel from other errors, and always
re-raise. Letting it propagate naturally through ``try/finally`` is the
normal pattern.
"""

from __future__ import annotations


class CancelRequested(BaseException):
    """Thrown into a plan generator by the engine to request cancellation.

    Inherits from :py:class:`BaseException` (not :py:class:`Exception`) so
    that broad ``except Exception:`` clauses in user plans do not
    accidentally swallow cancellation. Plans that need to distinguish
    cancellation from other errors should catch this explicitly and always
    re-raise.

    Attributes:
        reason: A short string describing why cancellation was requested
            (e.g., ``"user"``, ``"engine_shutdown"``, ``"keyboard_interrupt"``).

    """

    def __init__(self, reason: str = "cancel") -> None:
        super().__init__(reason)
        self.reason = reason


class PlanError(Exception):
    """Raised when a plan is malformed or its schema is invalid.

    Examples:
        - A plan decorated with ``run(...)`` neither passes explicit
          ``setpoints``/``measured`` args nor yields ``Describe`` first.
        - The descriptor contains two parameters sharing a ``register_name``.

    """
