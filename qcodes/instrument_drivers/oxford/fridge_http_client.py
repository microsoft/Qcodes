import aiohttp
import asyncio
import async_timeout

async def fetch(session, url):
    with async_timeout.timeout(10):
        async with session.get(url) as response:
            return await response.text()

async def set_value(session, url):
    with async_timeout.timeout(10):
        async with session.post(url) as response:
            return await response.text()

async def main(loop):
    async with aiohttp.ClientSession(loop=loop) as session:
        html = await fetch(session, url='http://127.0.0.1:8000/T1?attribute=value')
        print(html)

        html = await set_value(session, url='http://127.0.0.1:8000/pid_mode?setpoint=on')
        print(html)
        html = await fetch(session, url='http://127.0.0.1:8000/pid_mode?attribute=value')
        print(html)



if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(loop))