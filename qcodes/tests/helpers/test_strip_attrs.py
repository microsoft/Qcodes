import pytest

from typing import Any, Dict

from qcodes.utils.helpers import strip_attrs


class A:
    x = 5
    y = 6


class BadKeysDict(Dict[Any, Any]):
    def keys(self):
        raise RuntimeError('you can\'t have the keys!')


class NoDelDict(Dict[Any, Any]):
    def __delitem__(self, item):
        raise KeyError('get your hands off me!')


def test_normal():
    a = A()
    a.x = 15
    a.z = 25

    strip_attrs(a)

    assert a.x == 5
    assert not hasattr(a, 'z')
    assert a.y == 6


def test_pathological():
    # just make sure this never errors, since it's meant to be used
    # during deletion
    a = A()
    a.__dict__ = BadKeysDict()

    a.fruit = 'mango'
    with pytest.raises(RuntimeError):
        a.__dict__.keys()

    strip_attrs(a)
    # no error, but the attribute is still there
    assert a.fruit == 'mango'

    a = A()
    a.__dict__ = NoDelDict()
    s = 'can\'t touch this!'
    a.x = s

    assert a.x == s
    # not sure how this doesn't raise, but it doesn't.
    # with self.assertRaises(KeyError):
    #     del a.x

    strip_attrs(a)
    assert a.x == s
