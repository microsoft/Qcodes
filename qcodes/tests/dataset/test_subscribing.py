# Test some subscription scenarios
from typing import List, Tuple, Dict, Union, Any
from numbers import Number

import pytest
from numpy import ndarray
import logging

import qcodes
from qcodes.dataset.descriptions.param_spec import ParamSpecBase
from qcodes.dataset.descriptions.dependencies import InterDependencies_
from qcodes.dataset.sqlite.connection import atomic_transaction

from qcodes.tests.common import default_config
from qcodes.tests.common import retry_until_does_not_throw


log = logging.getLogger(__name__)

VALUE = Union[str, Number, List[Any], ndarray, bool]


class MockSubscriber():
    """
    A basic subscriber factory that creates a subscriber, that
    just puts results and length into state.
    *Important*
    This class is extremely dangerous! Within the callback,
    you cannot read or write to the database/dataset because it
    is called from another thread than the one holding the connection of the
    dataset!
    """
    def __init__(self, ds, lg):
        self.lg = lg
        self.ds = ds

    def __call__(self, results: List[Tuple[VALUE]],
                 length: int, state: Dict[Any, Any]) -> None:
        log.debug(f'got log {self.lg} and dataset {self.ds.completed}.')
        state[length] = results


def config_subscriber_factory(ds, l):
    def config_subscriber(results, length, state):
        state[length] = results
        log.debug(f'got log {l} and dataset {ds.completed}.')
    return config_subscriber


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


@pytest.mark.flaky(reruns=5)
@pytest.mark.serial
def test_basic_subscription(dataset, basic_subscriber):
    xparam = ParamSpecBase(name='x',
                           paramtype='numeric',
                           label='x parameter',
                           unit='V')
    yparam = ParamSpecBase(name='y',
                           paramtype='numeric',
                           label='y parameter',
                           unit='Hz')
    idps = InterDependencies_(dependencies={yparam: (xparam,)})
    dataset.set_interdependencies(idps)
    dataset.mark_started()

    sub_id = dataset.subscribe(basic_subscriber, min_wait=0, min_count=1,
                               state={})

    assert len(dataset.subscribers) == 1
    assert list(dataset.subscribers.keys()) == [sub_id]

    expected_state = {}

    for x in range(10):
        y = -x**2
        dataset.add_results([{'x': x, 'y': y}])
        expected_state[x+1] = [(x, y)]

        @retry_until_does_not_throw(
            exception_class_to_expect=AssertionError, delay=0.5, tries=10)
        def assert_expected_state():
            assert dataset.subscribers[sub_id].state == expected_state

        assert_expected_state()

    dataset.unsubscribe(sub_id)

    assert len(dataset.subscribers) == 0
    assert list(dataset.subscribers.keys()) == []

    # Ensure the trigger for the subscriber has been removed from the database
    get_triggers_sql = "SELECT * FROM sqlite_master WHERE TYPE = 'trigger';"
    triggers = atomic_transaction(
        dataset.conn, get_triggers_sql).fetchall()
    assert len(triggers) == 0


def test_subscription_from_config(dataset, basic_subscriber):
    """
    This test is similar to `test_basic_subscription`, with the only
    difference that another subscriber from a config file is added.
    """
    # This string represents the config file in the home directory:
    config = """
    {
        "subscription":{
            "subscribers":{
                "test_subscriber":{
                    "factory": "qcodes.tests.dataset.test_subscribing.MockSubscriber",
                    "factory_kwargs":{
                        "lg": false
                    },
                    "subscription_kwargs":{
                        "min_wait": 0,
                        "min_count": 1,
                        "callback_kwargs": {}
                    }
                }
            }
        }
    }
    """
    # This little dance around the db_location is due to the fact that the
    # dataset fixture creates a dataset in a db in a temporary directory.
    # Therefore we need to 'backup' the path to the db when using the
    # default configuration.
    db_location = qcodes.config.core.db_location
    with default_config(user_config=config):
        qcodes.config.core.db_location = db_location

        assert 'test_subscriber' in qcodes.config.subscription.subscribers

        xparam = ParamSpecBase(name='x',
                           paramtype='numeric',
                           label='x parameter',
                           unit='V')
        yparam = ParamSpecBase(name='y',
                              paramtype='numeric',
                              label='y parameter',
                              unit='Hz')
        idps = InterDependencies_(dependencies={yparam: (xparam,)})
        dataset.set_interdependencies(idps)

        dataset.mark_started()

        sub_id = dataset.subscribe(basic_subscriber, min_wait=0, min_count=1,
                                   state={})
        sub_id_c = dataset.subscribe_from_config('test_subscriber')
        assert len(dataset.subscribers) == 2
        assert list(dataset.subscribers.keys()) == [sub_id, sub_id_c]

        expected_state = {}

        # Here we are only testing 2 to reduce the CI time
        for x in range(2):
            y = -x**2
            dataset.add_results([{'x': x, 'y': y}])
            expected_state[x+1] = [(x, y)]

            @retry_until_does_not_throw(
                exception_class_to_expect=AssertionError, tries=10)
            def assert_expected_state():
                assert dataset.subscribers[sub_id].state == expected_state
                assert dataset.subscribers[sub_id_c].state == expected_state

            assert_expected_state()


def test_subscription_from_config_wrong_name(dataset):
    """
    This test checks that an exception is thrown if a wrong name for a
    subscriber is passed
    """
    # This string represents the config file in the home directory:
    config = """
    {
        "subscription":{
            "subscribers":{
                "test_subscriber_wrong":{
                    "factory": "qcodes.tests.dataset.test_subscribing.MockSubscriber",
                    "factory_kwargs":{
                        "lg": false
                    },
                    "subscription_kwargs":{
                        "min_wait": 0,
                        "min_count": 1,
                        "callback_kwargs": {}
                    }
                }
            }
        }
    }
    """
    db_location = qcodes.config.core.db_location
    with default_config(user_config=config):
        qcodes.config.core.db_location = db_location

        assert 'test_subscriber' not in qcodes.config.subscription.subscribers
        with pytest.raises(RuntimeError):
            sub_id_c = dataset.subscribe_from_config('test_subscriber')
