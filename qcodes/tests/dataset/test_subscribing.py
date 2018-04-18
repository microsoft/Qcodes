# Test some subscription scenarios
import tempfile
import os
from typing import List, Tuple, Dict, Union
from numbers import Number

import pytest
from numpy import ndarray

import qcodes as qc
import qcodes.dataset.sqlite_base as mut
from qcodes.dataset.param_spec import ParamSpec
from qcodes.dataset.database import initialise_database

VALUE = Union[str, Number, List, ndarray, bool]


@pytest.fixture(scope="function")
def empty_temp_db():
    # create a temp database for testing
    with tempfile.TemporaryDirectory() as tmpdirname:
        qc.config["core"]["db_location"] = os.path.join(tmpdirname, 'temp.db')
        qc.config["core"]["db_debug"] = True
        initialise_database()
        yield


@pytest.fixture(scope='function')
def experiment(empty_temp_db):
    e = qc.new_experiment("test-experiment", sample_name="test-sample")
    yield e
    e.conn.close()


@pytest.fixture(scope='function')
def dataset(experiment):
    dataset = qc.new_data_set("test-dataset")
    yield dataset
    dataset.unsubscribe_all()
    dataset.conn.close()


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

    for x in range(10):
        y = -x**2
        dataset.add_result({'x': x, 'y': y})
        expected_state[x+1] = [(x, y)]
        assert dataset.subscribers[sub_id].state == expected_state
