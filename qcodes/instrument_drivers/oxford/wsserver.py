
import asyncio
import websockets
import json
"""
This is a simple example of lisining for websocket data from a remote triton.
It requires a post message to first be sent to the Triton to the enablewebsockets endpoint.
An example of how this can be done is in enable_websockets.py
"""
async def getdata(websocket, path):

    while True:
        print("awaiting data")
        data = await websocket.recv()
        data = json.loads(data)
        print(data)


print("Starting server")
start_server = websockets.serve(getdata, 'localhost', 8765)
print("run event loop 1")
asyncio.get_event_loop().create_task(start_server)
asyncio.get_event_loop().run_forever()