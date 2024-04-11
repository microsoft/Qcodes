from dataclasses import dataclass

import pytest

from qcodes.utils import checked_getattr_indexed, getattr_indexed


@dataclass
class A:
    x: object
    y: object


@dataclass
class B:
    a_list: list
    b_dict: dict


def test_regular_unindexed() -> None:
    a = A(x=1, y=2)

    assert getattr_indexed(a, "x") == 1
    assert getattr_indexed(a, "y") == 2


def test_checked_regular_unindexed() -> None:
    a = A(x=1, y=2)

    assert checked_getattr_indexed(a, "x", int) == 1
    assert checked_getattr_indexed(a, "y", int) == 2


def test_checked_notype_unindexed() -> None:
    # just make sure this never errors, since it's meant to be used
    # during deletion
    a = A(x=1, y=2)

    with pytest.raises(TypeError):
        checked_getattr_indexed(a, "x", str)


def test_checked_notfound_unindexed() -> None:
    # just make sure this never errors, since it's meant to be used
    # during deletion
    a = A(x=1, y=2)

    with pytest.raises(AttributeError):
        getattr_indexed(a, "z")

    with pytest.raises(AttributeError):
        checked_getattr_indexed(a, "z", int)

    with pytest.raises(AttributeError):
        checked_getattr_indexed(a, "z", str)


def test_regular_indexed() -> None:
    a1 = A(x=1, y=2)
    a2 = A(x=3, y=4)
    b1 = B(a_list=[a1, a2], b_dict={2: [a1, a2]})

    assert getattr_indexed(b1, "a_list[0]") is a1
    assert getattr_indexed(b1, "a_list[1]") is a2
    assert getattr_indexed(b1, "b_dict[2][0]") is a1


def test_checked_indexed() -> None:
    a1 = A(x=1, y=2)
    a2 = A(x=3, y=4)
    b1 = B(a_list=[a1, a2], b_dict={2: [a1, a2]})

    assert checked_getattr_indexed(b1, "a_list[0]", A) is a1
    assert checked_getattr_indexed(b1, "a_list[1]", A) is a2
    assert checked_getattr_indexed(b1, "b_dict[2][0]", (A, B)) is a1


def test_notype_indexed() -> None:
    a1 = A(x=1, y=2)
    a2 = A(x=3, y=4)
    b1 = B(a_list=[a1, a2], b_dict={2: [a1, a2]})

    with pytest.raises(TypeError):
        checked_getattr_indexed(b1, "a_list[0]", B)
    with pytest.raises(TypeError):
        assert checked_getattr_indexed(b1, "a_list[1]", B) is a2
    with pytest.raises(TypeError):
        assert checked_getattr_indexed(b1, "b_dict[2][0]", B) is a1
