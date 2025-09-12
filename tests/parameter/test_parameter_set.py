from typing import TYPE_CHECKING

import pytest

from qcodes.parameters import ManualParameter, ParameterSet

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def manual_parameters() -> "Generator[tuple[ManualParameter, ...], None, None]":
    param1 = ManualParameter("param1")
    param2 = ManualParameter("param2")
    param3 = ManualParameter("param3")

    yield param1, param2, param3


def test_parameter_set_preserves_order(
    manual_parameters: tuple[ManualParameter, ...],
) -> None:
    param1, param2, param3 = manual_parameters
    param_set = ParameterSet((param1, param2, param3))
    param_list = list(param_set)
    assert param_list[0] is param1
    assert param_list[1] is param2
    assert param_list[2] is param3

    param_set = ParameterSet((param3, param1, param2))
    param_list = list(param_set)
    assert param_list[0] is param3
    assert param_list[1] is param1
    assert param_list[2] is param2

    param_set.clear()
    assert len(param_set) == 0
    param_set.update(ParameterSet((param1, param2, param3)))
    param_list = list(param_set)
    assert param_list[0] is param1
    assert param_list[1] is param2
    assert param_list[2] is param3

    param_set.clear()
    assert len(param_set) == 0
    param_set.add(param1)
    param_set.add(param2)
    param_set.add(param3)
    param_list = list(param_set)
    assert param_list[0] is param1
    assert param_list[1] is param2
    assert param_list[2] is param3

    param_set.remove(param2)
    param_list = list(param_set)
    assert param_list[0] is param1
    assert param_list[1] is param3

    param_set.add(param1)
    assert len(param_set) == 2
    param_list = list(param_set)
    assert param_list[0] is param1
    assert param_list[1] is param3


def test_parameter_set_operations(
    manual_parameters: tuple[ManualParameter, ...],
) -> None:
    param1, param2, param3 = manual_parameters
    param_set1 = ParameterSet((param1, param2, param3))
    param_set2 = ParameterSet((param3, param1, param2))
    param_subset1 = ParameterSet((param1, param2))

    assert param_set1 == param_set2  # Should this fail, because the order is different?
    assert param_subset1 < param_set1
    assert param_set1 > param_subset1
    assert param_subset1 <= param_set1
    assert param_set1 >= param_subset1

    union_set = ParameterSet([param1]) | ParameterSet([param2])
    assert len(union_set) == 2

    difference_set = param_set2 - ParameterSet([param1])
    assert len(difference_set) == 2

    intersection_set = ParameterSet([param1, param2]) & ParameterSet([param2, param3])
    assert len(intersection_set) == 1
    assert param2 in intersection_set
