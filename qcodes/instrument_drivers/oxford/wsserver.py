
import asyncio
import websockets
import json
import logging

import numpy as np

#myarray = tuple(np.zeros((10000)))
# logger = logging.getLogger('websockets')
# logger.setLevel(logging.DEBUG)
# logger.addHandler(logging.StreamHandler())

async def getdata(websocket, path):

    while True:
        print("awaiting data")
        data = await websocket.recv()
        data = json.loads(data)
        print(data)


async def start_sending(websocket, path):
    send_data = {}
    while True:
        send_data['enable'] = True
        print("enabling data")
        await websocket.send(json.dumps(send_data))
        await asyncio.sleep(5)
        print("disable data")
        send_data['enable'] = False
        await websocket.send(json.dumps(send_data))
        await asyncio.sleep(5)



print("Starting server")
start_server = websockets.serve(getdata, 'localhost', 8765)
start_server2 = websockets.serve(start_sending, 'localhost', 8766)
print("run event loop 1")
asyncio.get_event_loop().create_task(start_server)
print("run event loop 2")
asyncio.get_event_loop().create_task(start_server2)
asyncio.get_event_loop().run_forever()