#!/usr/bin/env python

import asyncio
import websockets

async def hello():
    async with websockets.connect('ws://localhost:8000/monitor') as websocket:
        i = 0
        while True:
            data = await websocket.recv()
            print(data)
            print("sending ping")
            pong = await websocket.ping()
            print("send ping")
            await pong
            print("got pong")
asyncio.get_event_loop().run_until_complete(hello())
