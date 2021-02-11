"""
Test attribute_set_to context manager
"""
from qcodes.utils.helpers import attribute_set_to


class X:
    y = 0


def test_attribute_set_to_value():
    """Test setting attribute to a value"""
    x = X()
    x.y = 1

    assert 1 == x.y

    with attribute_set_to(x, 'y', 2):
        assert 2 == x.y

    assert 1 == x.y


def test_attribute_set_to_object():
    """Test setting attribute to an object"""
    x = X()
    original_object = X()
    x.y = original_object

    assert original_object == x.y
    assert original_object is x.y

    new_object = X()
    with attribute_set_to(x, 'y', new_object):
        assert new_object == x.y
        assert new_object is x.y

    assert original_object == x.y
    assert original_object is x.y
