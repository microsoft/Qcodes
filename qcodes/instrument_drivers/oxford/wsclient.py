#!/usr/bin/env python

import asyncio
import websockets

async def hello():
    async with websockets.connect('ws://localhost:8000/ws') as websocket:
        i = 0
        while True:
            data = await websocket.recv()
            print(data)
asyncio.get_event_loop().run_until_complete(hello())
