import numpy as np
import pytest
from qcodes.utils.helpers import compare_dictionaries


def test_same():
    # NOTE(alexcjohnson): the numpy array and list compare equal,
    # even though a list and tuple would not. See TODO in
    # compare_dictionaries.
    a = {'a': 1, 2: [3, 4, {5: 6}], 'b': {'c': 'd'}, 'x': np.array([7, 8])}
    b = {'a': 1, 2: [3, 4, {5: 6}], 'b': {'c': 'd'}, 'x': [7, 8]}

    match, err = compare_dictionaries(a, b)
    assert match
    assert err == ''


def test_bad_dict():
    # NOTE(alexcjohnson):
    # this is a valid dict, but not good JSON because the tuple key cannot
    # be converted into a string.
    # It throws an error in compare_dictionaries, which is likely what we
    # want, but we should be aware of it.
    a = {(5, 6): (7, 8)}
    with pytest.raises(TypeError):
        compare_dictionaries(a, a)


def test_key_diff():
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


def test_val_diff_simple():
    a = {'a': 1}
    b = {'a': 2}

    match, err = compare_dictionaries(a, b)

    assert not match
    assert 'Value of "d1[a]" ("1", type"<class \'int\'>") not same as' in err
    assert '"d2[a]" ("2", type"<class \'int\'>")' in err


def test_val_diff_seq():
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


def test_nested_key_diff():
    a = {'a': {'b': 'c'}}
    b = {'a': {'d': 'c'}}

    match, err = compare_dictionaries(a, b)

    assert not match
    assert 'Key d1[a][b] not in d2' in err
    assert 'Key d2[a][d] not in d1' in err
