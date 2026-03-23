"""
Tests for qcodes.utils.deep_update_utils - recursive dict merging.
"""

from qcodes.utils.deep_update_utils import deep_update


def test_simple_key_value_update() -> None:
    """Test updating simple key-value pairs."""
    dest = {"a": 1, "b": 2}
    update = {"b": 3}
    result = deep_update(dest, update)
    assert result["a"] == 1
    assert result["b"] == 3


def test_nested_dict_merging() -> None:
    """Test that nested dicts are merged recursively."""
    dest = {"a": {"x": 1, "y": 2}}
    update = {"a": {"y": 3, "z": 4}}
    result = deep_update(dest, update)
    assert result["a"] == {"x": 1, "y": 3, "z": 4}


def test_deeply_nested_dict_merging() -> None:
    """Test recursive merging multiple levels deep."""
    dest = {"a": {"b": {"c": 1, "d": 2}}}
    update = {"a": {"b": {"d": 3, "e": 4}}}
    result = deep_update(dest, update)
    assert result["a"]["b"] == {"c": 1, "d": 3, "e": 4}


def test_lists_replaced_entirely() -> None:
    """Test that lists are replaced completely, not merged."""
    dest = {"a": [1, 2, 3]}
    update = {"a": [4, 5]}
    result = deep_update(dest, update)
    assert result["a"] == [4, 5]


def test_new_keys_added() -> None:
    """Test that new keys from update are added to dest."""
    dest = {"a": 1}
    update = {"b": 2, "c": 3}
    result = deep_update(dest, update)
    assert result == {"a": 1, "b": 2, "c": 3}


def test_non_dict_replaces_dict() -> None:
    """Test that a non-dict value replaces a dict value."""
    dest = {"a": {"x": 1}}
    update = {"a": "string_value"}
    result = deep_update(dest, update)
    assert result["a"] == "string_value"


def test_dict_replaces_non_dict() -> None:
    """Test that a dict value replaces a non-dict value."""
    dest = {"a": "string_value"}
    update = {"a": {"x": 1}}
    result = deep_update(dest, update)
    assert result["a"] == {"x": 1}


def test_returns_dest_dict() -> None:
    """Test that deep_update returns the dest dict (mutated in place)."""
    dest = {"a": 1}
    update = {"b": 2}
    result = deep_update(dest, update)
    assert result is dest


def test_deep_copy_of_update_values() -> None:
    """Test that mutations to the update dict don't affect dest."""
    inner_list = [1, 2, 3]
    dest: dict = {}
    update = {"a": inner_list}
    deep_update(dest, update)

    inner_list.append(4)
    assert dest["a"] == [1, 2, 3]


def test_deep_copy_of_nested_update_values() -> None:
    """Test that deep copies are made for nested structures."""
    inner_dict = {"x": [1, 2]}
    dest: dict = {"a": 1}
    update = {"b": inner_dict}
    deep_update(dest, update)

    inner_dict["x"].append(3)
    assert dest["b"]["x"] == [1, 2]


def test_empty_update() -> None:
    """Test that an empty update leaves dest unchanged."""
    dest = {"a": 1, "b": 2}
    result = deep_update(dest, {})
    assert result == {"a": 1, "b": 2}


def test_empty_dest() -> None:
    """Test updating an empty dest with values."""
    dest: dict = {}
    update = {"a": 1, "b": {"c": 2}}
    result = deep_update(dest, update)
    assert result == {"a": 1, "b": {"c": 2}}


def test_none_values() -> None:
    """Test that None values are handled correctly."""
    dest = {"a": 1}
    update = {"a": None}
    result = deep_update(dest, update)
    assert result["a"] is None
