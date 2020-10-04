import pytest

from qcodes.instrument.parameter import invert_val_mapping
from qcodes.utils.helpers import create_on_off_val_mapping


def test_values_of_mapping_are_only_the_given_two():
    val_mapping = create_on_off_val_mapping(on_val='666', off_val='000')
    values_set = set(list(val_mapping.values()))
    assert values_set == {'000', '666'}


def test_its_inverse_maps_only_to_booleans():
    inverse = invert_val_mapping(
        create_on_off_val_mapping(on_val='666', off_val='000'))

    assert inverse == {'666': True, '000': False}


@pytest.mark.parametrize(('on_val', 'off_val'),
                         ((1, 0),
                          (1.0, 0.0),
                          ('1', '0'),
                          (True, False)
                         ))
def test_create_on_off_val_mapping_for(on_val, off_val):
    """
    Explicitly test ``create_on_off_val_mapping`` function
    by covering some of the edge cases of ``on_val`` and ``off_val``
    """
    val_mapping = create_on_off_val_mapping(on_val=on_val,
                                            off_val=off_val)

    values_list = list(set(val_mapping.values()))

    assert len(values_list) == 2
    assert on_val in values_list
    assert off_val in values_list

    assert val_mapping[1] is on_val
    assert val_mapping[True] is on_val
    assert val_mapping['1'] is on_val
    assert val_mapping['ON'] is on_val
    assert val_mapping['On'] is on_val
    assert val_mapping['on'] is on_val

    assert val_mapping[0] is off_val
    assert val_mapping[False] is off_val
    assert val_mapping['0'] is off_val
    assert val_mapping['OFF'] is off_val
    assert val_mapping['Off'] is off_val
    assert val_mapping['off'] is off_val

    inverse = invert_val_mapping(val_mapping)

    assert inverse[on_val] is True
    assert inverse[off_val] is False
