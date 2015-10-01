#######################################################################
# This file is imported into tests only if the syntax is supported    #
#                                                                     #
# The package itself has been downgraded to only use py 3.3 syntax    #
# for asynchronous functions, but the tests will ensure that it works #
# with the newer py 3.5 syntax too                                    #
#######################################################################

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
