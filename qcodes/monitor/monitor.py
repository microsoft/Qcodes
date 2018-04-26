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

import asyncio
import logging
import os
import time
import json
import http.server
import socketserver
import webbrowser
import datetime
from copy import deepcopy
from contextlib import suppress

from threading import Thread
from typing import Dict, Any
from asyncio import CancelledError
import functools

import websockets

SERVER_PORT = 3000

log = logging.getLogger(__name__)


def _get_metadata(*parameters) -> Dict[str, Any]:
    """
    Return a dict that contains the parameter metadata grouped by the
    instrument it belongs to.
    """
    ts = time.time()
    # group meta data by instrument if any
    metas = {} # type: Dict
    for parameter in parameters:
        _meta = getattr(parameter, "_latest", None)
        if _meta:
            meta = deepcopy(_meta)
        else:
            raise ValueError("Input is not a parameter; Refusing to proceed")
        # convert to string
        meta['value'] = str(meta['value'])
        if isinstance(meta["ts"], datetime.datetime):
            meta["ts"] = time.mktime(meta["ts"].timetuple())
        meta["name"] = parameter.label or parameter.name
        meta["unit"] = parameter.unit
        # find the base instrument in case this is a channel parameter
        baseinst = parameter._instrument
        while hasattr(baseinst, '_parent'):
            baseinst = baseinst._parent
        accumulator = metas.get(str(baseinst), [])
        accumulator.append(meta)
        metas[str(baseinst)] = accumulator
    parameters_out = []
    for instrument in metas:
        temp = {"instrument": instrument, "parameters": metas[instrument]}
        parameters_out.append(temp)
    state = {"ts": ts, "parameters": parameters_out}
    return state


def _handler(parameters, interval: int):

    async def serverFunc(websocket, path):
        while True:
            try:
                try:
                    meta = _get_metadata(*parameters)
                except ValueError as e:
                    log.exception(e)
                    break
                log.debug(f"sending.. to {websocket}")
                try:
                    await websocket.send(json.dumps(meta))
                # mute browser disconnects
                except websockets.exceptions.ConnectionClosed as e:
                    log.debug(e)
                await asyncio.sleep(interval)
            except CancelledError:
                log.debug("Got CancelledError")
                break

        log.debug("Stopping Websocket handler")

    return serverFunc


class Monitor(Thread):
    running = None
    server = None

    def __init__(self, *parameters, interval=1):
        """
        Monitor qcodes parameters.

        Args:
            *parameters: Parameters to monitor
            interval: How often one wants to refresh the values
        """
        # let the thread start
        time.sleep(0.01)
        super().__init__()
        self.loop = None
        self._parameters = parameters
        self._monitor(*parameters, interval=interval)
        Monitor.running = self

    def run(self):
        """
        Start the event loop and run forever
        """
        log.debug("Running Websocket server")
        self.loop = asyncio.new_event_loop()
        self.loop_is_closed = False
        asyncio.set_event_loop(self.loop)
        try:
            server_start = websockets.serve(self.handler, '127.0.0.1', 5678)
            self.server = self.loop.run_until_complete(server_start)
            self.loop.run_forever()
        except OSError as e:
            # The code above may throw an OSError
            # if the socket cannot be bound
            log.exception(e)
        finally:
            log.debug("loop stopped")
            log.debug("Pending tasks at close: {}".format(
                asyncio.Task.all_tasks(self.loop)))
            self.loop.close()
            while not self.loop.is_closed():
                log.debug("waiting for loop to stop and close")
                time.sleep(0.01)
            self.loop_is_closed = True
            log.debug("loop closed")

    def update_all(self):
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
        log.debug("Pending tasks at stop: {}".format(asyncio.Task.all_tasks(self.loop)))
        self.loop.stop()

    def join(self, timeout=None) -> None:
        """
        Overwrite Thread.join to make sure server is stopped before
        joining avoiding a potential deadlock.
        """
        log.debug("Shutting down server")
        try:
            asyncio.run_coroutine_threadsafe(self.__stop_server(), self.loop)
        except RuntimeError as e:
            # the above may throw a runtime error if the loop is already
            # stopped in which case there is nothing more to do
            log.exception("Could not close loop")
        while not self.loop_is_closed:
            log.debug("waiting for loop to stop and close")
            time.sleep(0.01)
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

    def _monitor(self, *parameters, interval=1):
        self.handler = _handler(parameters, interval=interval)
        # TODO (giulioungaretti) read from config

        log.debug("Start monitoring thread")

        if Monitor.running:
            # stop the old server
            log.debug("Stopping and restarting server")
            Monitor.running.stop()

        self.start()

        # let the thread start
        time.sleep(0.01)
        log.debug("Start monitoring server")


class Server():

    def __init__(self, port=3000):
        self.port = port
        self.handler = http.server.SimpleHTTPRequestHandler
        self.httpd = socketserver.TCPServer(("", self.port), self.handler)
        self.static_dir = os.path.join(os.path.dirname(__file__), 'dist')

    def run(self):
        os.chdir(self.static_dir)
        log.debug("serving directory %s", self.static_dir)
        log.info("Open browser at http://localhost::{}".format(self.port))
        self.httpd.serve_forever()

    def stop(self):
        self.httpd.shutdown()


if __name__ == "__main__":
    server = Server(SERVER_PORT)
    print("Open browser at http://localhost:{}".format(server.port))
    try:
        webbrowser.open("http://localhost:{}".format(server.port))
        server.run()
    except KeyboardInterrupt:
        exit()
