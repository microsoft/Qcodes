#! /usr/bin/env python
# vim:fenc=utf-8
#
# Copyright © 2017 unga <giulioungaretti@me.com>
#
# Distributed under terms of the MIT license.
"""
Monitor a set of parameters in a background thread
stream output over websocket

To start monitor, run this file, or if qcodes is installed as a module:

``% python -m qcodes.monitor.monitor``

Add parameters to monitor in your measurement by creating a new monitor with a
list of parameters to monitor:

``monitor = qcodes.Monitor(param1, param2, param3, ...)``
"""


import sys
import logging
import os
import time
import json
from contextlib import suppress
from typing import Dict, Union, Any, Optional, Sequence, Callable, Awaitable, TYPE_CHECKING
from collections import defaultdict

import asyncio
from asyncio import CancelledError
from threading import Thread, Event

import socketserver
import webbrowser
import websockets

try:
    from websockets.legacy.server import serve
except ImportError:
    # fallback for websockets < 9
    # for the same reason we only support typechecking with websockets 9
    from websockets import serve  # type:ignore[attr-defined,no-redef]

if TYPE_CHECKING:
    from websockets.legacy.server import WebSocketServerProtocol, WebSocketServer

from qcodes.instrument.parameter import Parameter


WEBSOCKET_PORT = 5678
SERVER_PORT = 3000

log = logging.getLogger(__name__)


def _get_metadata(*parameters: Parameter) -> Dict[str, Any]:
    """
    Return a dictionary that contains the parameter metadata grouped by the
    instrument it belongs to.
    """
    metadata_timestamp = time.time()
    # group metadata by instrument
    metas: Dict[Any, Any] = defaultdict(list)
    for parameter in parameters:
        # Get the latest value from the parameter,
        # respecting the max_val_age parameter
        meta: Dict[str, Optional[Union[float, str]]] = {}
        meta["value"] = str(parameter.get_latest())
        timestamp = parameter.get_latest.get_timestamp()
        if timestamp is not None:
            meta["ts"] = timestamp.timestamp()
        else:
            meta["ts"] = None
        meta["name"] = parameter.label or parameter.name
        meta["unit"] = parameter.unit

        # find the base instrument that this parameter belongs to
        baseinst = parameter.root_instrument
        if baseinst is None:
            metas["Unbound Parameter"].append(meta)
        else:
            metas[str(baseinst)].append(meta)

    # Create list of parameters, grouped by instrument
    parameters_out = []
    for instrument in metas:
        temp = {"instrument": instrument, "parameters": metas[instrument]}
        parameters_out.append(temp)

    state = {"ts": metadata_timestamp, "parameters": parameters_out}
    return state


def _handler(parameters: Sequence[Parameter], interval: float) \
        -> Callable[["WebSocketServerProtocol", str], Awaitable[None]]:
    """
    Return the websockets server handler.
    """
    async def server_func(websocket: "WebSocketServerProtocol", _: str) -> None:
        """
        Create a websockets handler that sends parameter values to a listener
        every "interval" seconds.
        """
        while True:
            try:
                # Update the parameter values
                try:
                    meta = _get_metadata(*parameters)
                except ValueError:
                    log.exception("Error getting parameters")
                    break
                log.debug("sending.. to %r", websocket)
                await websocket.send(json.dumps(meta))
                # Wait for interval seconds and then send again
                await asyncio.sleep(interval)
            except (CancelledError, websockets.exceptions.ConnectionClosed):
                log.debug("Got CancelledError or ConnectionClosed",
                          exc_info=True)
                break
        log.debug("Closing websockets connection")

    return server_func


class Monitor(Thread):
    """
    QCodes Monitor - WebSockets server to monitor qcodes parameters.
    """
    running = None

    def __init__(self, *parameters: Parameter, interval: float = 1):
        """
        Monitor qcodes parameters.

        Args:
            *parameters: Parameters to monitor.
            interval: How often one wants to refresh the values.
        """
        super().__init__()

        # Check that all values are valid parameters
        for parameter in parameters:
            if not isinstance(parameter, Parameter):
                raise TypeError(f"We can only monitor QCodes "
                                f"Parameters, not {type(parameter)}")

        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.server: Optional["WebSocketServer"] = None
        self._parameters = parameters
        self.loop_is_closed = Event()
        self.server_is_started = Event()
        self.handler = _handler(parameters, interval=interval)

        log.debug("Start monitoring thread")
        if Monitor.running:
            # stop the old server
            log.debug("Stopping and restarting server")
            Monitor.running.stop()
        self.start()

        # Wait until the loop is running
        self.server_is_started.wait(timeout=5)
        if not self.server_is_started.is_set():
            raise RuntimeError("Failed to start server")
        Monitor.running = self

    def run(self) -> None:
        """
        Start the event loop and run forever.
        """
        log.debug("Running Websocket server")
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            server_start = serve(self.handler, '127.0.0.1',
                                            WEBSOCKET_PORT, close_timeout=1)
            self.server = self.loop.run_until_complete(server_start)
            self.server_is_started.set()
            self.loop.run_forever()
        except OSError:
            # The code above may throw an OSError
            # if the socket cannot be bound
            log.exception("Server could not be started")
        finally:
            log.debug("loop stopped")
            log.debug("Pending tasks at close: %r",
                      asyncio.all_tasks(self.loop))
            self.loop.close()
            log.debug("loop closed")
            self.loop_is_closed.set()

    def update_all(self) -> None:
        """
        Update all parameters in the monitor.
        """
        for parameter in self._parameters:
            # call get if it can be called without arguments
            with suppress(TypeError):
                parameter.get()

    def stop(self) -> None:
        """
        Shutdown the server, close the event loop and join the thread.
        Setting active Monitor to ``None``.
        """
        self.join()
        Monitor.running = None

    async def __stop_server(self) -> None:
        log.debug("asking server %r to close", self.server)
        if self.server is not None:
            self.server.close()
        log.debug("waiting for server to close")
        if self.loop is not None and self.server is not None:
            await self.loop.create_task(self.server.wait_closed())
        log.debug("stopping loop")
        if self.loop is not None:
            log.debug("Pending tasks at stop: %r",
                      asyncio.all_tasks(self.loop))
            self.loop.stop()

    def join(self, timeout: Optional[float] = None) -> None:
        """
        Overwrite ``Thread.join`` to make sure server is stopped before
        joining avoiding a potential deadlock.
        """
        log.debug("Shutting down server")
        if not self.is_alive():
            # we run this check before trying to run to prevent a cryptic
            # error message
            log.debug("monitor is dead")
            return
        try:
            if self.loop is not None:
                asyncio.run_coroutine_threadsafe(self.__stop_server(),
                                                 self.loop)
        except RuntimeError:
            # the above may throw a runtime error if the loop is already
            # stopped in which case there is nothing more to do
            log.exception("Could not close loop")
        self.loop_is_closed.wait(timeout=5)
        if not self.loop_is_closed.is_set():
            raise RuntimeError("Failed to join loop")
        log.debug("Loop reported closed")
        super().join(timeout=timeout)
        log.debug("Monitor Thread has joined")

    @staticmethod
    def show() -> None:
        """
        Overwrite this method to show/raise your monitor GUI
        F.ex.

        ::

            import webbrowser
            url = "localhost:3000"
            # Open URL in new window, raising the window if possible.
            webbrowser.open_new(url)

        """
        webbrowser.open(f"http://localhost:{SERVER_PORT}")


if __name__ == "__main__":
    import http.server
    # If this file is run, create a simple webserver that serves a simple
    # website that can be used to view monitored parameters.
    STATIC_DIR = os.path.join(os.path.dirname(__file__), 'dist')
    os.chdir(STATIC_DIR)
    try:
        log.info("Starting HTTP Server at http://localhost:%i", SERVER_PORT)
        with socketserver.TCPServer(("", SERVER_PORT),
                                    http.server.SimpleHTTPRequestHandler) as httpd:
            log.debug("serving directory %s", STATIC_DIR)
            webbrowser.open(f"http://localhost:{SERVER_PORT}")
            httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting Down HTTP Server")
