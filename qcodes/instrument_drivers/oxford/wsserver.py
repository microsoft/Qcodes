
import asyncio
import websockets
import json

async def getdata(websocket, path):
    print("awiting data")
    while True:
        data = await websocket.recv()
        data = json.loads(data)
        print(data)


print("Starting server")
start_server = websockets.serve(getdata, 'localhost', 8765)
print("run event loop")
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()