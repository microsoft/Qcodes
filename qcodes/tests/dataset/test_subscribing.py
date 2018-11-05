# Test some subscription scenarios
from typing import List, Tuple, Dict, Union
from numbers import Number

import pytest
from numpy import ndarray

from qcodes.dataset.param_spec import ParamSpec
# pylint: disable=unused-import
from qcodes.tests.dataset.temporary_databases import (empty_temp_db,
                                                      experiment,
                                                      dataset)
from qcodes.tests.common import retry_until_does_not_throw

VALUE = Union[str, Number, List, ndarray, bool]



@pytest.fixture(scope='function')
def basic_subscriber():
    """
    A basic subscriber that just puts results and length into
    state
    """

    def subscriber(results: List[Tuple[VALUE]], length: int,
                   state: Dict) -> None:
        state[length] = results

    return subscriber


def test_basic_subscription(dataset, basic_subscriber):
    xparam = ParamSpec(name='x', paramtype='numeric', label='x parameter',
                       unit='V')
    yparam = ParamSpec(name='y', paramtype='numeric', label='y parameter',
                       unit='Hz', depends_on=[xparam])
    dataset.add_parameter(xparam)
    dataset.add_parameter(yparam)

    sub_id = dataset.subscribe(basic_subscriber, min_wait=0, min_count=1,
                               state={})

    assert len(dataset.subscribers) == 1
    assert list(dataset.subscribers.keys()) == [sub_id]

    expected_state = {}

    # Here we are only testing 2 to reduce the CI time
    for x in range(2):
        y = -x**2
        dataset.add_result({'x': x, 'y': y})
        expected_state[x+1] = [(x, y)]

        @retry_until_does_not_throw(
            exception_class_to_expect=AssertionError, delay=0, tries=10)
        def assert_expected_state():
            assert dataset.subscribers[sub_id].state == expected_state

        assert_expected_state()
