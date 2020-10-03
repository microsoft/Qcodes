from unittest import TestCase

import pytest
import numpy as np

from qcodes.utils.helpers import (compare_dictionaries,
                                  partial_with_docstring)
from qcodes.utils.helpers import attribute_set_to


class A:
    x = 5
    y = 6


class BadKeysDict(dict):
    def keys(self):
        raise RuntimeError('you can\'t have the keys!')


class NoDelDict(dict):
    def __delitem__(self, item):
        raise KeyError('get your hands off me!')


class TestCompareDictionaries(TestCase):
    def test_same(self):
        # NOTE(alexcjohnson): the numpy array and list compare equal,
        # even though a list and tuple would not. See TODO in
        # compare_dictionaries.
        a = {'a': 1, 2: [3, 4, {5: 6}], 'b': {'c': 'd'}, 'x': np.array([7, 8])}
        b = {'a': 1, 2: [3, 4, {5: 6}], 'b': {'c': 'd'}, 'x': [7, 8]}

        match, err = compare_dictionaries(a, b)
        assert match
        assert err == ''

    def test_bad_dict(self):
        # NOTE(alexcjohnson):
        # this is a valid dict, but not good JSON because the tuple key cannot
        # be converted into a string.
        # It throws an error in compare_dictionaries, which is likely what we
        # want, but we should be aware of it.
        a = {(5, 6): (7, 8)}
        with pytest.raises(TypeError):
            compare_dictionaries(a, a)

    def test_key_diff(self):
        a = {'a': 1, 'c': 4}
        b = {'b': 1, 'c': 4}

        match, err = compare_dictionaries(a, b)

        assert not match
        assert 'Key d1[a] not in d2' in err
        assert 'Key d2[b] not in d1' in err

        # try again with dict names for completeness
        match, err = compare_dictionaries(a, b, 'a', 'b')

        assert not match
        assert 'Key a[a] not in b' in err
        assert 'Key b[b] not in a' in err

    def test_val_diff_simple(self):
        a = {'a': 1}
        b = {'a': 2}

        match, err = compare_dictionaries(a, b)

        assert not match
        assert 'Value of "d1[a]" ("1", type"<class \'int\'>") not same as' in err
        assert '"d2[a]" ("2", type"<class \'int\'>")' in err

    def test_val_diff_seq(self):
        # NOTE(alexcjohnson):
        # we don't dive recursively into lists at the moment.
        # Perhaps we want to? Seems like list equality does a deep comparison,
        # so it's not necessary to get ``match`` right, but the error message
        # could be more helpful if we did.
        a = {'a': [1, {2: 3}, 4]}
        b = {'a': [1, {5: 6}, 4]}

        match, err = compare_dictionaries(a, b)

        assert not match
        assert 'Value of "d1[a]" ("[1, {2: 3}, 4]", ' \
                      'type"<class \'list\'>") not same' in err
        assert '"d2[a]" ("[1, {5: 6}, 4]", type"<class \'list\'>")' in \
                      err

    def test_nested_key_diff(self):
        a = {'a': {'b': 'c'}}
        b = {'a': {'d': 'c'}}

        match, err = compare_dictionaries(a, b)

        assert not match
        assert 'Key d1[a][b] not in d2' in err
        assert 'Key d2[a][d] not in d1' in err


class TestAttributeSetToContextManager(TestCase):
    """
    Test attribute_set_to context manager
    """

    class X:
        y = 0

    def test_attribute_set_to_value(self):
        """Test setting attribute to a value"""
        x = self.X()
        x.y = 1

        assert 1 == x.y

        with attribute_set_to(x, 'y', 2):
            assert 2 == x.y

        assert 1 == x.y

    def test_attribute_set_to_object(self):
        """Test setting attribute to an object"""
        x = self.X()
        original_object = self.X()
        x.y = original_object

        assert original_object == x.y
        assert original_object is x.y

        new_object = self.X()
        with attribute_set_to(x, 'y', new_object):
            assert new_object == x.y
            assert new_object is x.y

        assert original_object == x.y
        assert original_object is x.y


class TestPartialWithDocstring(TestCase):
    """Test the sane partial function"""
    def test_main(self):
        def f():
            pass

        docstring = "some docstring"
        g = partial_with_docstring(f, docstring)
        assert g.__doc__ == docstring
