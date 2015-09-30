from unittest import TestCase
import os
from datetime import datetime, timedelta

from qcodes.utils.helpers import (is_function, is_sequence, permissive_range,
                                  wait_secs, make_unique)


class TestIsFunction(TestCase):
    def test_non_function(self):
        self.assertFalse(is_function(0, 0))
        self.assertFalse(is_function('hello!', 0))
        self.assertFalse(is_function(None, 0))

    def test_function(self):
        def f0():
            raise RuntimeError('function should not get called')

        def f1(a):
            raise RuntimeError('function should not get called')

        def f2(a, b):
            raise RuntimeError('function should not get called')

        self.assertTrue(is_function(f0, 0))
        self.assertTrue(is_function(f1, 1))
        self.assertTrue(is_function(f2, 2))

        self.assertFalse(is_function(f0, 1) or is_function(f0, 2))
        self.assertFalse(is_function(f1, 0) or is_function(f1, 2))
        self.assertFalse(is_function(f2, 0) or is_function(f2, 1))

        # make sure we only accept valid arg_count
        with self.assertRaises(TypeError):
            is_function(f0, 'lots')
        with self.assertRaises(TypeError):
            is_function(f0, -1)

    class AClass(object):
        def method_a(self):
            raise RuntimeError('function should not get called')

        def method_b(self, v):
            raise RuntimeError('function should not get called')

        async def method_c(self, v):
            raise RuntimeError('function should not get called')

    def test_methods(self):
        a = self.AClass()
        self.assertTrue(is_function(a.method_a, 0))
        self.assertFalse(is_function(a.method_a, 1))
        self.assertTrue(is_function(a.method_b, 1))
        self.assertTrue(is_function(a.method_c, 1, coroutine=True))

    def test_type_cast(self):
        self.assertTrue(is_function(int, 1))
        self.assertTrue(is_function(float, 1))
        self.assertTrue(is_function(str, 1))

        self.assertFalse(is_function(int, 0) or is_function(int, 2))
        self.assertFalse(is_function(float, 0) or is_function(float, 2))
        self.assertFalse(is_function(str, 0) or is_function(str, 2))

    def test_coroutine_check(self):
        def f_sync():
            raise RuntimeError('function should not get called')

        async def f_async():
            raise RuntimeError('function should not get called')

        self.assertTrue(is_function(f_sync, 0))
        self.assertTrue(is_function(f_sync, 0, coroutine=False))
        self.assertTrue(is_function(f_async, 0, coroutine=True))
        self.assertFalse(is_function(f_async, 0))
        self.assertFalse(is_function(f_async, 0, coroutine=False))


class TestIsSequence(TestCase):
    def a_func():
        raise RuntimeError('this function shouldn\'t get called')

    class AClass():
        pass

    def test_yes(self):
        f = open(os.listdir('.')[0], 'r')
        yes_sequence = [
            [],
            [1, 2, 3],
            range(5),

            # we do have to be careful about generators...
            # ie don't call len() or iterate twice
            (i**2 for i in range(5)),

            # and some possibly useless or confusing matches
            # that we might want to rule out:
            set((1, 2, 3)),
            {1: 2, 3: 4},
            f
        ]

        for val in yes_sequence:
            self.assertTrue(is_sequence(val))

        f.close()

    def test_no(self):
        no_sequence = [
            1,
            1.0,
            True,
            None,
            'you can iterate a string but we won\'t',
            b'nor will we iterate bytes',
            self.a_func,
            self.AClass,
            self.AClass()
        ]

        for val in no_sequence:
            self.assertFalse(is_sequence(val))


class TestPermissiveRange(TestCase):
    def test_bad_calls(self):
        bad_args = [
            [],
            [1],
            [1, 2],
            [None, 1, .1],
            [1, None, .1],
            [1, 2, 'not too far']
        ]

        for args in bad_args:
            with self.assertRaises(Exception):
                permissive_range(*args)

    def test_good_calls(self):
        good_args = {
            (1, 7, 2): [1, 3, 5],
            (1, 7, 4): [1, 5],
            (1, 7, 7): [1],
            (1.0, 7, 2): [1.0, 3.0, 5.0],
            (1, 7.0, 2): [1.0, 3.0, 5.0],
            (1, 7, 2.0): [1.0, 3.0, 5.0],
            (1.0, 7.0, 2.0): [1.0, 3.0, 5.0],
            (1.0, 7.000000001, 2.0): [1.0, 3.0, 5.0, 7.0],
            (1, 7, -2): [1, 3, 5],
            (7, 1, 2): [7, 5, 3],
            (1.0, 7.0, -2.0): [1.0, 3.0, 5.0],
            (7.0, 1.0, 2.0): [7.0, 5.0, 3.0],
            (7.0, 1.0, -2.0): [7.0, 5.0, 3.0],
            (1.5, 1.8, 0.1): [1.5, 1.6, 1.7]
        }

        for args, result in good_args.items():
            self.assertEqual(permissive_range(*args), result)


class TestWaitSecs(TestCase):
    def test_bad_calls(self):
        bad_args = [None, 1, 1.0, True]
        for arg in bad_args:
            with self.assertRaises(TypeError):
                wait_secs(arg)

    def test_good_calls(self):
        for secs in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]:
            finish_datetime = datetime.now() + timedelta(seconds=secs)
            secs_out = wait_secs(finish_datetime)
            self.assertGreater(secs_out, secs - 1e-4)
            self.assertLessEqual(secs_out, secs)

    def test_warning(self):
        # TODO: how to test what logging is doing?
        secs_out = wait_secs(datetime.now() - timedelta(seconds=1))
        self.assertEqual(secs_out, 0)


class TestMakeUnique(TestCase):
    def test_no_changes(self):
        for s, existing in (('a', []), ('a', {}), ('a', ('A', ' a', 'a '))):
            self.assertEqual(make_unique(s, existing), s)

    def test_changes(self):
        self.assertEqual(make_unique('a', ('a',)), 'a_2')
        self.assertEqual(make_unique('a_2', ('a_2',)), 'a_2_2')
        self.assertEqual(make_unique('a', ('a', 'a_2', 'a_3')), 'a_4')
