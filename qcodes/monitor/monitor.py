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

from threading import Thread
from typing import Dict
from concurrent.futures import Future
from concurrent.futures import CancelledError
import functools

import websockets

SERVER_PORT = 3000

log = logging.getLogger(__name__)


def _get_metadata(*parameters) -> Dict[float, list]:
    """
    Return a dict that contains the paraemter metadata grouped by the
    instrument it belongs to.
    """
    ts = time.time()
    # group meta data by instrument if any
    metas = {}
    for parameter in parameters:
        _meta = getattr(parameter, "_latest", None)
        if _meta:
            meta = _meta()
        else:
            raise ValueError("Input is not a paraemter; Refusing to proceed")
        # convert to string
        meta['value'] = str(meta['value'])
        if meta["ts"] is not None:
            meta["ts"] = time.mktime(meta["ts"].timetuple())
        meta["name"] = parameter.label or parameter.name
        meta["unit"] = parameter.unit
        accumulator = metas.get(str(parameter._instrument), [])
        accumulator.append(meta)
        metas[str(parameter._instrument)] = accumulator
    parameters = []
    for instrument in metas:
        temp = {"instrument": instrument, "parameters": metas[instrument]}
        parameters.append(temp)
    state = {"ts": ts, "parameters": parameters}
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
                log.debug("sending..")
                try:
                    await websocket.send(json.dumps(meta))
                # mute browser discconects
                except websockets.exceptions.ConnectionClosed as e:
                    log.debug(e)
                    pass
                await asyncio.sleep(interval)
            except CancelledError:
                break
        log.debug("closing sever")

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
        super().__init__()
        self.loop = None
        # start the server to server monitor http/files
        if Monitor.server:
            self.show()
        else:
            Monitor.server = Server(port=SERVER_PORT)
            Monitor.server.start()
        self._monitor(*parameters, interval=1)

    def run(self):
        """
        Start the event loop and run forever
        """
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        Monitor.running = self
        self.show()
        self.loop.run_forever()

    def stop(self):
        """
        Shutodwn the server, close the event loop and join the thread
        """
        # this contains the server
        # or any exception
        server = self.future_restult.result()
        # server.close()
        self.loop.call_soon_threadsafe(server.close)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.join()
        Monitor.running = None

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
        handler = _handler(parameters, interval=interval)
        # TODO (giulioungaretti) read from config
        server = websockets.serve(handler, '127.0.0.1', 5678)

        log.debug("Start monitoring thread")

        if Monitor.running:
            # stop the old server
            log.debug("Stoppging and restarting server")
            Monitor.running.stop()

        self.start()

        # let the thread start
        time.sleep(0.001)

        log.debug("Start monitoring server")
        self._add_task(server)

    def _create_task(self, future, coro):
        task = self.loop.create_task(coro)
        future.set_result(task)

    def _add_task(self, coro):
        future = Future()
        self.task = coro
        p = functools.partial(self._create_task, future, coro)
        self.loop.call_soon_threadsafe(p)
        # this stores the result of the future
        self.future_restult = future.result()
        self.future_restult.add_done_callback(_log_result)


def _log_result(future):
    try:
        future.result()
        log.debug("Started server loop")
    except:
        log.exception("Could not start server loop")


class Server(Thread):

    def __init__(self, port=3000):
        self.port = port
        self.handler = http.server.SimpleHTTPRequestHandler
        self.httpd = socketserver.TCPServer(("", self.port), self.handler)
        self.static_dir = os.path.join(os.path.dirname(__file__), 'dist')
        super().__init__()

    def run(self):
        os.chdir(self.static_dir)
        log.debug("serving directory %s", self.static_dir)
        log.debug("serving at port", self.port)
        self.httpd.serve_forever()

    def stop(self):
        self.httpd.shutdown()
        self.join()
