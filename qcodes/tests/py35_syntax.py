import asyncio


# f_async is for test_helpers
async def f_async():
    raise RuntimeError('function should not get called')


# async1, 2, 3_new are for test_sync_async
async def async1_new(v):
    return v**2


async def async2_new(v, n):
    for i in range(n):
        await asyncio.sleep(0.001)
    return v**2


async def async3_new(v, n):
    return await async2_new(v, n)
