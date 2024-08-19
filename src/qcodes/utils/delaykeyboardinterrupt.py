import logging
import signal
import threading
from collections.abc import Callable
from types import FrameType, TracebackType
from typing import TYPE_CHECKING, Any, cast

from typing_extensions import deprecated

from qcodes.utils.deprecate import QCoDeSDeprecationWarning

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from opentelemetry.util.types import (  # pyright: ignore[reportMissingTypeStubs]
        AttributeValue,
    )
log = logging.getLogger(__name__)


class DelayedKeyboardInterrupt:
    signal_received: tuple[int, FrameType | None] | None = None
    # the handler type is seemingly only defined in typesheeed so copy it here
    # manually https://github.com/python/typeshed/blob/main/stdlib/signal.pyi
    old_handler: "Callable[[int, FrameType | None], Any] | int | None" = None

    def __init__(self, context: "Mapping[str, AttributeValue] | None" = None) -> None:
        """
        A context manager to wrap a piece of code to ensure that a
        KeyboardInterrupt is not triggered by a SIGINT during the execution of
        this context. A second SIGINT will trigger the KeyboardInterrupt
        immediately.

        Inspired by https://stackoverflow.com/questions/842557/how-to-prevent-a-block-of-code-from-being-interrupted-by-keyboardinterrupt-in-py

        Args:
            context: A dictionary of attributes to be logged when the
                KeyboardInterrupt is triggered. This is useful for debugging
                where the interrupt was triggered. The attributes should be compatible
                with OpenTelemetry's Attributes type.

        """
        self._context = context if context is not None else {}

    def __enter__(self) -> None:
        is_main_thread = threading.current_thread() is threading.main_thread()
        is_default_sig_handler = (
            signal.getsignal(signal.SIGINT) is signal.default_int_handler
        )
        if is_default_sig_handler and is_main_thread:
            self.old_handler = signal.signal(signal.SIGINT, self.handler)
        elif is_default_sig_handler:
            log.debug(
                "Not on main thread cannot intercept interrupts", extra=self._context
            )

    def handler(self, sig: int, frame: FrameType | None) -> None:
        def create_forceful_handler(
            context: "Mapping[str, AttributeValue]",
        ) -> "Callable[[int, FrameType | None], None]":
            def forceful_handler(sig: int, frame: FrameType | None) -> None:
                print(
                    "Second SIGINT received. Triggering KeyboardInterrupt immediately."
                )
                log.info(
                    "Second SIGINT received. Triggering KeyboardInterrupt immediately.",
                    extra=context,
                )
                # The typing of signals seems to be inconsistent
                # since handlers must be types to take an optional frame
                # but default_int_handler does not take None.
                # see https://github.com/python/typeshed/pull/6599
                frame = cast(FrameType, frame)
                signal.default_int_handler(sig, frame)

            return forceful_handler

        self.signal_received = (sig, frame)
        print(
            "Received SIGINT, Will interrupt at first suitable time. "
            "Send second SIGINT to interrupt immediately."
        )
        # now that we have gotten one SIGINT make the signal
        # trigger a keyboard interrupt normally
        forceful_handler = create_forceful_handler(self._context)

        signal.signal(signal.SIGINT, forceful_handler)
        log.info("SIGINT received. Delaying KeyboardInterrupt.", extra=self._context)

    @deprecated(
        "forceful_handler is no longer part of the public api of DelayedKeyboardInterrupt",
        category=QCoDeSDeprecationWarning,
    )
    @staticmethod
    def forceful_handler(sig: int, frame: FrameType | None) -> None:
        print("Second SIGINT received. Triggering KeyboardInterrupt immediately.")
        log.info(
            "Second SIGINT received. Triggering KeyboardInterrupt immediately.",
        )
        # The typing of signals seems to be inconsistent
        # since handlers must be types to take an optional frame
        # but default_int_handler does not take None.
        # see https://github.com/python/typeshed/pull/6599
        frame = cast(FrameType, frame)
        signal.default_int_handler(sig, frame)

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.old_handler is not None:
            signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received is not None:
            if self.old_handler is not None and not isinstance(self.old_handler, int):
                self.old_handler(*self.signal_received)
