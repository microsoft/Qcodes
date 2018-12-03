#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 unga <giulioungaretti@me.com>
#
# Distributed under terms of the MIT license.
"""
Monitor a set of parameter in a background thread
stream opuput over websocket
"""

import logging
import os
import time
import json
from contextlib import suppress
from typing import Dict, Any
from collections import defaultdict

import asyncio
from asyncio import CancelledError
from threading import Thread, Event

import http.server
import socketserver
import webbrowser
import websockets

from qcodes.instrument.parameter import Parameter

WEBSOCKET_PORT = 5678
SERVER_PORT = 3000

log = logging.getLogger(__name__)


def _get_metadata(*parameters) -> Dict[str, Any]:
    """
    Return a dict that contains the parameter metadata grouped by the
    instrument it belongs to.
    """
    ts = time.time()
    # group metadata by instrument
    metas = defaultdict(list) # type: dict
    for parameter in parameters:
        # Check we have a list of parameters
        if not isinstance(parameter, Parameter):
            raise ValueError("{} is not a parameter".format(parameter))

        # Get the latest value from the parameter, respecting the max_val_age parameter
        meta = {}
        meta["value"] = str(parameter.get_latest())
        if parameter.get_latest.get_timestamp() is not None:
            meta["ts"] = parameter.get_latest.get_timestamp().timestamp()
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

    state = {"ts": ts, "parameters": parameters_out}
    return state


def _handler(parameters, interval: int):
    """
    Return the websockets server handler
    """
    async def server_func(websocket, path):
        """
        Create a websockets handler that sends parameter values to a listener
        every "interval" seconds
        """
        while True:
            try:
                # Update the parameter values
                try:
                    meta = _get_metadata(*parameters)
                except ValueError as e:
                    log.exception(e)
                    break
                log.debug("sending.. to %r", websocket)
                await websocket.send(json.dumps(meta))
                # Wait for interval seconds and then send again
                await asyncio.sleep(interval)
            except (CancelledError, websockets.exceptions.ConnectionClosed):
                log.debug("Got CancelledError or ConnectionClosed")
                break
        log.debug("Closing websockets connection")

    return server_func

class Monitor(Thread):
    """
    QCodes Monitor - WebSockets server to monitor qcodes parameters
    """
    running = None

    def __init__(self, *parameters, interval=1):
        """
        Monitor qcodes parameters.

        Args:
            *parameters: Parameters to monitor
            interval: How often one wants to refresh the values
        """
        super().__init__()
        self.loop = None
        self.server = None
        self._parameters = parameters
        self.loop_is_closed = Event()
        self.handler = _handler(parameters, interval=interval)

        log.debug("Start monitoring thread")
        if Monitor.running:
            # stop the old server
            log.debug("Stopping and restarting server")
            Monitor.running.stop()
        self.start()

        # Wait until the loop is running
        while self.loop is None:
            time.sleep(0.01)
        Monitor.running = self

    def run(self):
        """
        Start the event loop and run forever
        """
        log.debug("Running Websocket server")
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            server_start = websockets.serve(self.handler, '127.0.0.1', WEBSOCKET_PORT)
            self.server = self.loop.run_until_complete(server_start)
            self.loop.run_forever()
        except OSError:
            # The code above may throw an OSError
            # if the socket cannot be bound
            log.exception("Server could not be started")
        finally:
            log.debug("loop stopped")
            log.debug("Pending tasks at close: %r",
                      asyncio.Task.all_tasks(self.loop))
            self.loop.close()
            log.debug("loop closed")
            self.loop_is_closed.set()

    def update_all(self):
        """
        Update all parameters in the monitor
        """
        for p in self._parameters:
            # call get if it can be called without arguments
            with suppress(TypeError):
                p.get()

    def stop(self) -> None:
        """
        Shutdown the server, close the event loop and join the thread.
        Setting active Monitor to None
        """
        self.join()
        Monitor.running = None

    async def __stop_server(self):
        log.debug("asking server to close")
        self.server.close()
        log.debug("waiting for server to close")
        await self.loop.create_task(self.server.wait_closed())
        log.debug("stopping loop")
        log.debug("Pending tasks at stop: %r",
                  asyncio.Task.all_tasks(self.loop))
        self.loop.stop()

    def join(self, timeout=None) -> None:
        """
        Overwrite Thread.join to make sure server is stopped before
        joining avoiding a potential deadlock.
        """
        log.debug("Shutting down server")
        if not self.is_alive():
            # we run this check before trying to run to prevent a cryptic
            # error message
            log.debug("monitor is dead")
            return
        try:
            asyncio.run_coroutine_threadsafe(self.__stop_server(), self.loop)
        except RuntimeError:
            # the above may throw a runtime error if the loop is already
            # stopped in which case there is nothing more to do
            log.exception("Could not close loop")
        self.loop_is_closed.wait()
        log.debug("Loop reported closed")
        super().join(timeout=timeout)
        log.debug("Monitor Thread has joined")

    @staticmethod
    def show():
        """
        Overwrite this method to show/raise your monitor GUI
        F.ex.

        ::

            import webbrowser
            url = "localhost:3000"
            # Open URL in new window, raising the window if possible.
            webbrowser.open_new(url)

        """
        webbrowser.open("http://localhost:{}".format(SERVER_PORT))

class Server():

    def __init__(self, port=SERVER_PORT):
        self.port = port
        self.handler = http.server.SimpleHTTPRequestHandler
        self.httpd = socketserver.TCPServer(("", self.port), self.handler)
        self.static_dir = os.path.join(os.path.dirname(__file__), 'dist')

    def run(self):
        os.chdir(self.static_dir)
        log.debug("serving directory %s", self.static_dir)
        self.httpd.serve_forever()

    def stop(self):
        self.httpd.shutdown()


if __name__ == "__main__":
    server = Server()
    print("Open browser at http://localhost:{}".format(server.port))
    try:
        webbrowser.open("http://localhost:{}".format(server.port))
        server.run()
    except KeyboardInterrupt:
        exit()
