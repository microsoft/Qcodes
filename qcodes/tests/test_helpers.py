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



class TestPartialWithDocstring(TestCase):
    """Test the sane partial function"""
    def test_main(self):
        def f():
            pass

        docstring = "some docstring"
        g = partial_with_docstring(f, docstring)
        assert g.__doc__ == docstring
