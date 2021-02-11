import pytest
from qcodes.utils.helpers import is_function


def test_non_function():
    assert not is_function(0, 0)
    assert not is_function('hello!', 0)
    assert not is_function(None, 0)


def test_function():
    def f0():
        raise RuntimeError('function should not get called')

    def f1(a):
        raise RuntimeError('function should not get called')

    def f2(a, b):
        raise RuntimeError('function should not get called')

    assert is_function(f0, 0)
    assert is_function(f1, 1)
    assert is_function(f2, 2)

    assert not (is_function(f0, 1) or is_function(f0, 2))
    assert not (is_function(f1, 0) or is_function(f1, 2))
    assert not (is_function(f2, 0) or is_function(f2, 1))

    # make sure we only accept valid arg_count
    with pytest.raises(TypeError):
        is_function(f0, 'lots')
    with pytest.raises(TypeError):
        is_function(f0, -1)


class AClass:
    def method_a(self):
        raise RuntimeError('function should not get called')

    def method_b(self, v):
        raise RuntimeError('function should not get called')

    async def method_c(self, v):
        raise RuntimeError('function should not get called')


def test_methods():
    a = AClass()
    assert is_function(a.method_a, 0)
    assert not is_function(a.method_a, 1)
    assert is_function(a.method_b, 1)
    assert is_function(a.method_c, 1, coroutine=True)


def test_type_cast():
    assert is_function(int, 1)
    assert is_function(float, 1)
    assert is_function(str, 1)

    assert not (is_function(int, 0) or is_function(int, 2))
    assert not (is_function(float, 0) or is_function(float, 2))
    assert not (is_function(str, 0) or is_function(str, 2))


def test_coroutine_check():
    def f_sync():
        raise RuntimeError('function should not get called')

    assert is_function(f_sync, 0)
    assert is_function(f_sync, 0, coroutine=False)

    async def f_async():
        raise RuntimeError('function should not get called')

    assert not is_function(f_async, 0, coroutine=False)
    assert is_function(f_async, 0, coroutine=True)
    assert not is_function(f_async, 0)
