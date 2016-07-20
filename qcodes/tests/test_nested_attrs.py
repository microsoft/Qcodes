from unittest import TestCase
from qcodes.utils.nested_attrs import NestedAttrAccess


class TestNestedAttrAccess(TestCase):
    def test_simple(self):
        obj = NestedAttrAccess()

        # before setting attr1
        self.assertEqual(obj.getattr('attr1', 99), 99)
        with self.assertRaises(AttributeError):
            obj.getattr('attr1')

        with self.assertRaises(TypeError):
            obj.setattr('attr1')

        self.assertFalse(hasattr(obj, 'attr1'))

        # set it to a value
        obj.setattr('attr1', 98)
        self.assertTrue(hasattr(obj, 'attr1'))

        self.assertEqual(obj.getattr('attr1', 99), 98)
        self.assertEqual(obj.getattr('attr1'), 98)

        # then delete it
        obj.delattr('attr1')

        with self.assertRaises(AttributeError):
            obj.delattr('attr1')

        with self.assertRaises(AttributeError):
            obj.getattr('attr1')

        # make and call a method
        def f(a, b=0):
            return a + b

        obj.setattr('m1', f)
        self.assertEqual(obj.callattr('m1', 4, 1), 5)
        self.assertEqual(obj.callattr('m1', 21, b=42), 63)

    def test_nested(self):
        obj = NestedAttrAccess()

        self.assertFalse(hasattr(obj, 'd1'))

        with self.assertRaises(TypeError):
            obj.setattr('d1')

        # set one attribute that creates nesting
        obj.setattr('d1', {'a': {1: 2, 'l': [5, 6]}})

        # can't nest inside a non-container
        with self.assertRaises(TypeError):
            obj.setattr('d1["a"][1]["secret"]', 42)

        # get the whole dict
        self.assertEqual(obj.getattr('d1'), {'a': {1: 2, 'l': [5, 6]}})
        self.assertEqual(obj.getattr('d1', 55), {'a': {1: 2, 'l': [5, 6]}})

        # get parts
        self.assertEqual(obj.getattr('d1["a"]'), {1: 2, 'l': [5, 6]})
        self.assertEqual(obj.getattr('d1["a"][1]'), 2)
        self.assertEqual(obj.getattr('d1["a"][1]', 3), 2)
        with self.assertRaises(KeyError):
            obj.getattr('d1["b"]')

        # add an attribute inside, then delete it again
        obj.setattr('d1["a"][2]', 4)
        self.assertEqual(obj.getattr('d1'), {'a': {1: 2, 2: 4, 'l': [5, 6]}})
        obj.delattr('d1["a"][2]')
        self.assertEqual(obj.getattr('d1'), {'a': {1: 2, 'l': [5, 6]}})
        self.assertEqual(obj.d1, {'a': {1: 2, 'l': [5, 6]}})

        # list access
        obj.setattr('d1["a"]["l"][0]', 7)
        obj.callattr('d1["a"]["l"].extend', [5, 3])
        obj.delattr('d1["a"]["l"][1]')
        # while we're at it test single quotes
        self.assertEqual(obj.getattr("d1['a']['l'][1]"), 5)
        self.assertEqual(obj.d1['a']['l'], [7, 5, 3])

    def test_bad_attr(self):
        obj = NestedAttrAccess()
        obj.d = {}
        # this one works
        obj.setattr('d["x"]', 1)

        bad_attrs = [
            '', '.', '[', 'x.', '[]',  # simply malformed
            '.x'  # don't put a dot at the start
            '["hi"]',  # can't set an item at the top level
            'd[x]', 'd["x]', 'd["x\']'  # quoting errors
        ]

        for attr in bad_attrs:
            with self.subTest(attr=attr):
                with self.assertRaises(ValueError):
                    obj.setattr(attr, 1)
