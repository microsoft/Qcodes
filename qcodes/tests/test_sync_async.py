import asyncio
from unittest import TestCase
from time import time
import sys

from qcodes.utils.sync_async import (wait_for_async, mock_async, mock_sync,
                                     syncable_command, NoCommandError)


class TestAsync(TestCase):
    async def async1(self, v):
        return v**2

    async def async2(self, v, n):
        for i in range(n):
            await asyncio.sleep(0.001)
        return v**2

    async def async3(self, v, n):
        return await self.async2(v, n)

    def test_simple(self):
        self.assertEqual(wait_for_async(self.async1, 2), 4)

    def test_await(self):
        t1 = time()
        self.assertEqual(wait_for_async(self.async2, 3, 100), 9)
        t2 = time()
        self.assertGreaterEqual(t2 - t1, 0.1)
        # measure of how good async timing is
        # answer: about a fraction of a millisecond, and always
        # longer than specified, never shorter
        # TODO: make some benchmarks so we can understand this on
        # different systems where it's deployed
        self.assertLess(t2 - t1, 0.2)

    def test_chain(self):
        t1 = time()
        self.assertEqual(wait_for_async(self.async3, 5, 100), 25)
        t2 = time()
        self.assertGreaterEqual(t2 - t1, 0.1)
        self.assertLess(t2 - t1, 0.2)

    def test_mock_async(self):
        def sync_no_args():
            return 42

        def sync_args(a=1, b=2, c=3):
            return a + b + c

        # we can't wait on a sync function
        with self.assertRaises(TypeError):
            self.assertEqual(wait_for_async(sync_no_args), 12)

        # but with mock_async we can
        # also tests argument passing in wait_for_async
        self.assertEqual(wait_for_async(mock_async(sync_no_args)), 42)
        self.assertEqual(wait_for_async(mock_async(sync_args)), 6)
        self.assertEqual(wait_for_async(mock_async(sync_args), 10), 15)
        self.assertEqual(wait_for_async(mock_async(sync_args), 0, 0), 3)
        self.assertEqual(wait_for_async(mock_async(sync_args), 2, 3, 4), 9)
        self.assertEqual(wait_for_async(mock_async(sync_args), b=10), 14)
        self.assertEqual(wait_for_async(mock_async(sync_args), 5, c=20), 27)

        with self.assertRaises(TypeError):
            wait_for_async(mock_async(sync_no_args), 0)
        with self.assertRaises(TypeError):
            wait_for_async(mock_async(sync_no_args), a=0)

    def test_mock_sync(self):
        f1 = mock_sync(self.async1)
        f2 = mock_sync(self.async2)
        f3 = mock_sync(self.async3)

        self.assertEqual(f1(6), 36)
        self.assertEqual(f2(7, n=10), 49)
        self.assertEqual(f3(8, 5), 64)

    def test_reentrant_wait_for_async(self):
        f = mock_sync(mock_async(mock_sync(self.async1)))
        self.assertEqual(f(9), 81)


class CustomError(Exception):
    pass


class TestSyncableCommand(TestCase):
    def test_bad_calls(self):
        with self.assertRaises(TypeError):
            syncable_command()

        with self.assertRaises(TypeError):
            syncable_command(cmd='')

        with self.assertRaises(TypeError):
            syncable_command(0, '', parse_function=lambda: 1)

        with self.assertRaises(TypeError):
            syncable_command(0, cmd='', exec_str='not a function')

        with self.assertRaises(TypeError):
            syncable_command(0, cmd='', aexec_str='also not a function')

    def test_no_cmd(self):
        with self.assertRaises(NoCommandError):
            syncable_command(0)

        def no_cmd_function():
            raise CustomError('no command')

        no_cmd, no_acmd = syncable_command(0, no_cmd_function=no_cmd_function)
        with self.assertRaises(CustomError):
            no_cmd()
        with self.assertRaises(CustomError):
            wait_for_async(no_acmd)

    def test_cmd_str(self):
        def f_now(x):
            return x + ' now'

        async def adelay(x):
            await asyncio.sleep(0.001)
            return x + ' delayed'

        def upper(s):
            return s.upper()

        # only sync exec_str
        cmd, acmd = syncable_command(0, 'pickles', exec_str=f_now)
        self.assertEqual(cmd(), 'pickles now')
        self.assertEqual(wait_for_async(acmd), 'pickles now')

        # only async exec_str
        cmd, acmd = syncable_command(0, 'lemons', aexec_str=adelay)
        self.assertEqual(cmd(), 'lemons delayed')
        self.assertEqual(wait_for_async(acmd), 'lemons delayed')

        # separate sync and async exec_str
        cmd, acmd = syncable_command(0, 'herring', exec_str=f_now,
                                     aexec_str=adelay)
        self.assertEqual(cmd(), 'herring now')
        self.assertEqual(wait_for_async(acmd), 'herring delayed')
        with self.assertRaises(TypeError):
            cmd(12)
        with self.assertRaises(TypeError):
            wait_for_async(acmd, 21)

        # separate sync/async with parsing
        cmd, acmd = syncable_command(0, 'blue', exec_str=f_now,
                                     aexec_str=adelay, parse_function=upper)
        self.assertEqual(cmd(), 'BLUE NOW')
        self.assertEqual(wait_for_async(acmd), 'BLUE DELAYED')

        # parameter insertion
        cmd, acmd = syncable_command(3, '{} is {:.2f}% better than {}',
                                     exec_str=f_now, aexec_str=adelay)
        self.assertEqual(cmd('ice cream', 56.2, 'cake'),
                         'ice cream is 56.20% better than cake now')
        self.assertEqual(wait_for_async(acmd, 'cheese', 30, 'cheese'),
                         'cheese is 30.00% better than cheese delayed')
        with self.assertRaises(ValueError):
            cmd('cake', 'a whole lot', 'pie')

        with self.assertRaises(TypeError):
            cmd('donuts', 100, 'bagels', 'with cream cheese')

    def test_cmd_function(self):
        def myexp(a, b):
            return a ** b

        async def mymod(a, b):
            await asyncio.sleep(0.001)
            return a % b

        # only sync
        cmd, acmd = syncable_command(2, myexp)
        self.assertEqual(cmd(10, 3), 1000)
        self.assertEqual(wait_for_async(acmd, 2, 10), 1024)
        with self.assertRaises(TypeError):
            syncable_command(3, myexp)
        with self.assertRaises(TypeError):
            syncable_command(2, mymod)

        # only async
        cmd, acmd = syncable_command(2, None, mymod)
        self.assertEqual(cmd(10, 3), 1)
        self.assertEqual(wait_for_async(acmd, 2, 10), 2)
        with self.assertRaises(TypeError):
            syncable_command(3, None, mymod)
        with self.assertRaises(TypeError):
            syncable_command(2, None, myexp)

        # both sync and async
        cmd, acmd = syncable_command(2, myexp, mymod)
        self.assertEqual(cmd(10, 3), 1000)
        self.assertEqual(wait_for_async(acmd, 10, 3), 1)
        with self.assertRaises(TypeError):
            cmd(1, 2, 3)
        with self.assertRaises(TypeError):
            wait_for_async(acmd, 4, 5, 6)
