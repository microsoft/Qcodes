from typing import ClassVar

import pytest

from qcodes.utils import DelegateAttributes


def test_delegate_dict() -> None:
    class ToDict(DelegateAttributes):
        delegate_attr_dicts: ClassVar[list[str]] = ["d"]
        apples = "green"

    td = ToDict()
    # td.d doesn't exist yet
    with pytest.raises(AttributeError):
        td.d

    # but you can still get other attributes
    assert td.apples == 'green'

    d = {'apples': 'red', 'oranges': 'orange'}
    td.d = d  # type: ignore[attr-defined]

    # you can get the whole dict still
    assert td.d == d

    # class attributes override the dict
    assert td.apples == 'green'

    # instance attributes do too
    td.apples = 'rotten'
    assert td.apples == 'rotten'

    # other dict attributes come through and can be added on the fly
    assert td.oranges == 'orange'
    d['bananas'] = 'yellow'
    assert td.bananas == 'yellow'

    # missing items still raise AttributeError, not KeyError
    with pytest.raises(AttributeError):
        td.kiwis

    # all appropriate items are in dir() exactly once
    for attr in ['apples', 'oranges', 'bananas']:
        assert dir(td).count(attr) == 1


def test_delegate_dicts() -> None:
    class ToDicts(DelegateAttributes):
        delegate_attr_dicts: ClassVar[list[str]] = ["d", "e"]

    td = ToDicts()
    e = {"cats": 12, "dogs": 3}
    td.e = e  # type: ignore[attr-defined]

    # you can still access the second one when the first doesn't exist
    with pytest.raises(AttributeError):
        td.d
    assert td.e == e
    assert td.cats == 12

    # the first beats out the second
    td.d = {"cats": 42, "chickens": 1000}  # type: ignore[attr-defined]
    assert td.cats == 42

    # but you can still access things only in the second
    assert td.dogs == 3

    # all appropriate items are in dir() exactly once
    for attr in ['cats', 'dogs', 'chickens']:
        assert dir(td).count(attr) == 1


def test_delegate_object() -> None:
    class Recipient:
        black = '#000'
        white = '#fff'

    class ToObject(DelegateAttributes):
        delegate_attr_objects: ClassVar[list[str]] = ["recipient"]
        gray = "#888"

    to_obj = ToObject()
    recipient = Recipient()

    # recipient not connected yet but you can look at other attributes
    with pytest.raises(AttributeError):
        to_obj.recipient
    assert to_obj.gray == '#888'

    to_obj.recipient = recipient  # type: ignore[attr-defined]

    # now you can access recipient through to_obj
    assert to_obj.black == '#000'

    # to_obj overrides but you can still access other recipient attributes
    #  "soft" black
    to_obj.black = "#444"  # type: ignore[attr-defined]
    assert to_obj.black == "#444"
    assert to_obj.white == "#fff"

    # all appropriate items are in dir() exactly once
    for attr in ['black', 'white', 'gray']:
        assert dir(to_obj).count(attr) == 1


def test_delegate_objects() -> None:
    class R1:
        a = 1
        b = 2
        c = 3

    class R2:
        a = 4
        b = 5
        d = 6

    class ToObjects(DelegateAttributes):
        delegate_attr_objects: ClassVar[list[str]] = ["r1", "r2"]
        a = 0
        e = 7
        r1 = R1()
        r2 = R2()

    to_objs = ToObjects()

    # main object overrides recipients
    assert to_objs.a == 0
    assert to_objs.e == 7

    # first object overrides second
    assert to_objs.b == 2
    assert to_objs.c == 3

    # second object gets the rest
    assert to_objs.d == 6

    # missing attributes still raise correctly
    with pytest.raises(AttributeError):
        to_objs.f

    # all appropriate items are in dir() exactly once
    for attr in 'abcde':
        assert dir(to_objs).count(attr) == 1


def test_delegate_both() -> None:
    class Recipient:
        rock = 0
        paper = 1
        scissors = 2

    my_recipient_dict = {'paper': 'Petta et al.', 'year': 2005}

    class ToBoth(DelegateAttributes):
        delegate_attr_objects: ClassVar[list[str]] = ["recipient_object"]
        delegate_attr_dicts: ClassVar[list[str]] = ["recipient_dict"]
        rock = "Eiger"
        water = "Lac Leman"
        recipient_dict = my_recipient_dict
        recipient_object = Recipient()

    tb = ToBoth()

    # main object overrides recipients
    assert tb.rock == 'Eiger'
    assert tb.water == 'Lac Leman'

    # dict overrides object
    assert tb.paper == 'Petta et al.'
    assert tb.year == 2005

    # object comes last
    assert tb.scissors == 2

    # missing attributes still raise correctly
    with pytest.raises(AttributeError):
        tb.ninja

    # all appropriate items are in dir() exactly once
    for attr in ['rock', 'paper', 'scissors', 'year', 'water']:
        assert dir(tb).count(attr) == 1
