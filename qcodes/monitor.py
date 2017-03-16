#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 unga <giulioungaretti@me.com>
#
# Distributed under terms of the MIT license.

"""
Monitor a set of parameter
"""
import asyncio
import logging
import time
import json

from threading import Thread
from concurrent.futures import Future
from concurrent.futures import CancelledError
import functools

import qcodes as qc
import websockets

log = logging.getLogger(__name__)


def _get_metadata(*parameters: qc.Parameter)-> dict:
    ts = time.time()
    metas = []
    for parameter in parameters:
        meta = parameter._latest()
        if meta["ts"] is not None:
            meta["ts"] = time.mktime(meta["ts"].timetuple())
        meta["name"] = parameter.label or parameter.name
        meta["unit"] = parameter.unit
        metas.append(meta)
    state = {"ts": ts, "parameters": metas}
    return state


def poll_and_send(parameters: qc.Parameter, interval: int):

    async def poll_and_send(websocket, path):
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

    return poll_and_send


def monitor(parameters: qc.Parameter, interval=1) -> list:
    handler = poll_and_send(parameters, interval=interval)
    server = websockets.serve(handler, '127.0.0.1', 5678)
    log.debug("Start monitoring thread")
    if AsyncThread.running:
        # stop the old server
        log.debug("stoppging and restarting server")
        AsyncThread.running.stop()
    thread = AsyncThread()
    thread.start()
    # let the thread start
    time.sleep(0.001)
    log.debug("Start monitoring server")
    thread.add_task(server)
    return thread


class AsyncThread(Thread):
    running = None

    def __init__(self):
        super().__init__()
        self.loop = None

    def run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        AsyncThread.running = self
        self.loop.run_forever()

    def stop(self):
        time.sleep(1)
        # this contains the server
        # or any exception
        server = self.future_restult.result()
        server.close()
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.join()
        AsyncThread.running = None

    def _add_task(self, future, coro):
        task = self.loop.create_task(coro)
        future.set_result(task)

    def add_task(self, coro):
        future = Future()
        self.task = coro
        p = functools.partial(self._add_task, future, coro)
        self.loop.call_soon_threadsafe(p)
        # this stores the result of the future
        self.future_restult = future.result()
        self.future_restult.add_done_callback(log_result)


def log_result(future):
    try:
        future.result()
        log.debug("started server loop")
    except:
        log.exception("Could not start")
