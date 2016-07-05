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
