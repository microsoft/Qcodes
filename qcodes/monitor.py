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
import time
import json

from threading import Thread
from typing import Dict
from concurrent.futures import Future
from concurrent.futures import CancelledError
import functools

from qcodes import Parameter
import qcodes as qc
import websockets

log = logging.getLogger(__name__)


def _get_metadata(*parameters: Parameter) -> Dict[float, list]:
    """
    Return a dict that contains the paraemter metadata grouped by the
    instrument it belongs to.
    """
    ts = time.time()
    # group meta data by instrument if any
    metas = {}
    for parameter in parameters:
        meta = parameter._latest()
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


def _handler(parameters: qc.Parameter, interval: int):

    async def serverFunc(websocket, path):
        while True:
            try:
                meta = _get_metadata(*parameters)
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

    def __init__(self, *parameters: qc.Parameter, interval=1):
        """
        Monitor qcodes parameters.

        Args:
            *parameters: Parameters to monitor
            interval: How often one wants to refresh the values
        """
        super().__init__()
        self.loop = None
        self._monitor(*parameters, interval=1)

    def run(self):
        """
        Start the event loop and run forever
        """
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        Monitor.running = self
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

    def _add_task(self, future, coro):
        task = self.loop.create_task(coro)
        future.set_result(task)

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
        raise NotImplemented

    def _monitor(self, *parameters: qc.Parameter, interval=1):
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
        self.add_task(server)

    def add_task(self, coro):
        future = Future()
        self.task = coro
        p = functools.partial(self._add_task, future, coro)
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
