from unittest import TestCase
import time
from datetime import datetime
import asyncio

from qcodes.utils.helpers import (is_function, is_sequence, permissive_range,
                                  wait_secs, make_unique, DelegateAttributes,
                                  LogCapture)


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

        @asyncio.coroutine
        def method_c(self, v):
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

        self.assertTrue(is_function(f_sync, 0))
        self.assertTrue(is_function(f_sync, 0, coroutine=False))

        # support pre-py3.5 async syntax
        @asyncio.coroutine
        def f_async_old():
            raise RuntimeError('function should not get called')

        self.assertFalse(is_function(f_async_old, 0, coroutine=False))
        self.assertTrue(is_function(f_async_old, 0, coroutine=True))
        self.assertFalse(is_function(f_async_old, 0))

        # test py3.5 syntax async functions
        try:
            from qcodes.tests.py35_syntax import f_async
            py35 = True
        except:
            py35 = False

        if py35:
            self.assertFalse(is_function(f_async, 0, coroutine=False))
            self.assertTrue(is_function(f_async, 0, coroutine=True))
            self.assertFalse(is_function(f_async, 0))


class TestIsSequence(TestCase):
    def a_func():
        raise RuntimeError('this function shouldn\'t get called')

    class AClass():
        pass

    def test_yes(self):
        f = open(__file__, 'r')
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
        bad_args = [None, datetime.now()]
        for arg in bad_args:
            with self.assertRaises(TypeError):
                wait_secs(arg)

    def test_good_calls(self):
        for secs in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]:
            finish_clock = time.perf_counter() + secs
            secs_out = wait_secs(finish_clock)
            self.assertGreater(secs_out, secs - 1e-4)
            self.assertLessEqual(secs_out, secs)

    def test_warning(self):
        with LogCapture() as s:
            secs_out = wait_secs(time.perf_counter() - 1)
        self.assertEqual(secs_out, 0)

        logstr = s.getvalue()
        s.close()
        self.assertEqual(logstr.count('negative delay'), 1, logstr)


class TestMakeUnique(TestCase):
    def test_no_changes(self):
        for s, existing in (('a', []), ('a', {}), ('a', ('A', ' a', 'a '))):
            self.assertEqual(make_unique(s, existing), s)

    def test_changes(self):
        self.assertEqual(make_unique('a', ('a',)), 'a_2')
        self.assertEqual(make_unique('a_2', ('a_2',)), 'a_2_2')
        self.assertEqual(make_unique('a', ('a', 'a_2', 'a_3')), 'a_4')


class TestDelegateAttributes(TestCase):
    def test_delegate_dict(self):
        class ToDict(DelegateAttributes):
            delegate_attr_dicts = ['d']
            apples = 'green'

        td = ToDict()
        # td.d doesn't exist yet
        with self.assertRaises(AttributeError):
            td.d

        # but you can still get other attributes
        self.assertEqual(td.apples, 'green')

        d = {'apples': 'red', 'oranges': 'orange'}
        td.d = d

        # you can get the whole dict still
        self.assertEqual(td.d, d)

        # class attributes override the dict
        self.assertEqual(td.apples, 'green')

        # instance attributes do too
        td.apples = 'rotten'
        self.assertEqual(td.apples, 'rotten')

        # other dict attributes come through and can be added on the fly
        self.assertEqual(td.oranges, 'orange')
        d['bananas'] = 'yellow'
        self.assertEqual(td.bananas, 'yellow')

        # missing items still raise AttributeError, not KeyError
        with self.assertRaises(AttributeError):
            td.kiwis

        # all appropriate items are in dir() exactly once
        for attr in ['apples', 'oranges', 'bananas']:
            self.assertEqual(dir(td).count(attr), 1)

    def test_delegate_dicts(self):
        class ToDicts(DelegateAttributes):
            delegate_attr_dicts = ['d', 'e']

        td = ToDicts()
        e = {'cats': 12, 'dogs': 3}
        td.e = e

        # you can still access the second one when the first doesn't exist
        with self.assertRaises(AttributeError):
            td.d
        self.assertEqual(td.e, e)
        self.assertEqual(td.cats, 12)

        # the first beats out the second
        td.d = {'cats': 42, 'chickens': 1000}
        self.assertEqual(td.cats, 42)

        # but you can still access things only in the second
        self.assertEqual(td.dogs, 3)

        # all appropriate items are in dir() exactly once
        for attr in ['cats', 'dogs', 'chickens']:
            self.assertEqual(dir(td).count(attr), 1)

    def test_delegate_object(self):
        class Recipient:
            black = '#000'
            white = '#fff'

        class ToObject(DelegateAttributes):
            delegate_attr_objects = ['recipient']
            gray = '#888'

        to_obj = ToObject()
        recipient = Recipient()

        # recipient not connected yet but you can look at other attributes
        with self.assertRaises(AttributeError):
            to_obj.recipient
        self.assertEqual(to_obj.gray, '#888')

        to_obj.recipient = recipient

        # now you can access recipient through to_obj
        self.assertEqual(to_obj.black, '#000')

        # to_obj overrides but you can still access other recipient attributes
        to_obj.black = '#444'  # "soft" black
        self.assertEqual(to_obj.black, '#444')
        self.assertEqual(to_obj.white, '#fff')

        # all appropriate items are in dir() exactly once
        for attr in ['black', 'white', 'gray']:
            self.assertEqual(dir(to_obj).count(attr), 1)

    def test_delegate_objects(self):
        class R1:
            a = 1
            b = 2
            c = 3

        class R2:
            a = 4
            b = 5
            d = 6

        class ToObjects(DelegateAttributes):
            delegate_attr_objects = ['r1', 'r2']
            a = 0
            e = 7
            r1 = R1()
            r2 = R2()

        to_objs = ToObjects()

        # main object overrides recipients
        self.assertEqual(to_objs.a, 0)
        self.assertEqual(to_objs.e, 7)

        # first object overrides second
        self.assertEqual(to_objs.b, 2)
        self.assertEqual(to_objs.c, 3)

        # second object gets the rest
        self.assertEqual(to_objs.d, 6)

        # missing attributes still raise correctly
        with self.assertRaises(AttributeError):
            to_objs.f

        # all appropriate items are in dir() exactly once
        for attr in 'abcde':
            self.assertEqual(dir(to_objs).count(attr), 1)

    def test_delegate_both(self):
        class Recipient:
            rock = 0
            paper = 1
            scissors = 2

        my_recipient_dict = {'paper': 'Petta et al.', 'year': 2005}

        class ToBoth(DelegateAttributes):
            delegate_attr_objects = ['recipient_object']
            delegate_attr_dicts = ['recipient_dict']
            rock = 'Eiger'
            water = 'Lac Leman'
            recipient_dict = my_recipient_dict
            recipient_object = Recipient()

        tb = ToBoth()

        # main object overrides recipients
        self.assertEqual(tb.rock, 'Eiger')
        self.assertEqual(tb.water, 'Lac Leman')

        # dict overrides object
        self.assertEqual(tb.paper, 'Petta et al.')
        self.assertEqual(tb.year, 2005)

        # object comes last
        self.assertEqual(tb.scissors, 2)

        # missing attributes still raise correctly
        with self.assertRaises(AttributeError):
            tb.ninja

        # all appropriate items are in dir() exactly once
        for attr in ['rock', 'paper', 'scissors', 'year', 'water']:
            self.assertEqual(dir(tb).count(attr), 1)
