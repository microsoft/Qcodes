# Test some subscription scenarios
import logging
from numbers import Number
from typing import TYPE_CHECKING, Any

import pytest
from numpy import ndarray

import qcodes
from qcodes.dataset import new_data_set
from qcodes.dataset.descriptions.dependencies import InterDependencies_
from qcodes.dataset.sqlite.connection import atomic_transaction
from qcodes.parameters import ParamSpecBase
from tests.common import retry_until_does_not_throw

if TYPE_CHECKING:
    from collections.abc import Callable
if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from qcodes.dataset.data_set import DataSet

log = logging.getLogger(__name__)

VALUE = str | Number | list[Any] | ndarray | bool


class MockSubscriber:
    """
    A basic subscriber factory that creates a subscriber, that
    just puts results and length into state.
    *Important*
    This class is extremely dangerous! Within the callback,
    you cannot read or write to the database/dataset because it
    is called from another thread than the one holding the connection of the
    dataset!
    """

    def __init__(self, ds: "DataSet", lg: "bool") -> None:
        self.lg = lg
        self.ds = ds

    def __call__(
        self, results: list[tuple[VALUE]], length: int, state: dict[Any, Any]
    ) -> None:
        log.debug(f"got log {self.lg} and dataset {self.ds.completed}.")
        state[length] = results


def config_subscriber_factory(
    ds: "DataSet", log: "logging.Logger"
) -> "Callable[[list[tuple[VALUE]], int, dict | None], None]":
    def config_subscriber(
        results: "list[tuple[VALUE]]", length: "int", state: "dict | None"
    ) -> "None":
        assert state is not None
        state[length] = results
        log.debug(f"got log {log} and dataset {ds.completed}.")

    return config_subscriber


@pytest.fixture(scope="function")
def basic_subscriber() -> "Callable[[list[tuple[VALUE]], int, dict | None], None]":
    """
    A basic subscriber that just puts results and length into
    state
    """

    def subscriber(
        results: list[tuple[VALUE]], length: int, state: dict | None
    ) -> None:
        assert state is not None
        state[length] = results

    return subscriber


@pytest.fixture(name="working_subscriber_config")
def _make_working_subscriber_config(tmp_path: "Path") -> "Generator[None, None, None]":
    # This string represents the config file in the home directory:
    config = """
    {
        "subscription":{
            "subscribers":{
                "test_subscriber":{
                    "factory": "tests.dataset.test_subscribing.MockSubscriber",
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
    tmp_config_file_path = tmp_path / "qcodesrc.json"
    with open(tmp_config_file_path, "w") as f:
        f.write(config)
    qcodes.config.update_config(str(tmp_path))
    yield


@pytest.fixture(name="broken_subscriber_config")
def _make_broken_subscriber_config(tmp_path: "Path") -> "Generator[None, None, None]":
    # This string represents the config file in the home directory:
    config = """
    {
        "subscription":{
            "subscribers":{
                "test_subscriber_wrong":{
                    "factory": "tests.dataset.test_subscribing.MockSubscriber",
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
    tmp_config_file_path = tmp_path / "qcodesrc.json"
    with open(tmp_config_file_path, "w") as f:
        f.write(config)
    qcodes.config.update_config(str(tmp_path))
    yield


def test_basic_subscription(
    dataset: "DataSet",
    basic_subscriber: "Callable[[list[tuple[VALUE]], int, dict | None], None]",
) -> None:
    xparam = ParamSpecBase(name="x", paramtype="numeric", label="x parameter", unit="V")
    yparam = ParamSpecBase(
        name="y", paramtype="numeric", label="y parameter", unit="Hz"
    )
    idps = InterDependencies_(dependencies={yparam: (xparam,)})
    dataset.set_interdependencies(idps)
    dataset.mark_started()

    sub_id = dataset.subscribe(basic_subscriber, min_wait=0, min_count=1, state={})

    assert len(dataset.subscribers) == 1
    assert list(dataset.subscribers.keys()) == [sub_id]

    expected_state = {}

    for x in range(10):
        y = -(x**2)
        dataset.add_results([{"x": x, "y": y}])
        expected_state[x + 1] = [(x, y)]

        @retry_until_does_not_throw(
            exception_class_to_expect=AssertionError, delay=0.05, tries=100
        )
        def assert_expected_state() -> "None":
            assert dataset.subscribers[sub_id].state == expected_state

        assert_expected_state()

    dataset.unsubscribe(sub_id)

    assert len(dataset.subscribers) == 0
    assert list(dataset.subscribers.keys()) == []

    # Ensure the trigger for the subscriber has been removed from the database
    get_triggers_sql = "SELECT * FROM sqlite_master WHERE TYPE = 'trigger';"
    triggers = atomic_transaction(dataset.conn, get_triggers_sql).fetchall()
    assert len(triggers) == 0


def test_subscribe_with_background_writing_not_supported(dataset: "DataSet") -> None:
    xparam = ParamSpecBase(name="x", paramtype="numeric")
    yparam = ParamSpecBase(name="y", paramtype="numeric")
    dataset.set_interdependencies(InterDependencies_(dependencies={yparam: (xparam,)}))
    dataset.mark_started(start_bg_writer=True)
    try:
        with pytest.raises(NotImplementedError, match="written in the background"):
            dataset.subscribe(lambda results, length, state: None)
        assert dataset.subscribers == {}
    finally:
        # completing the dataset shuts down the background writer
        dataset.mark_completed()


def test_subscribe_while_another_run_writes_in_background(
    dataset: "DataSet",
) -> None:
    """
    Only the run that is written in the background can not be subscribed to.
    Another dataset in the same database can, and can then be written in the
    main thread once the background run has completed.
    """
    xparam = ParamSpecBase(name="x", paramtype="numeric")
    yparam = ParamSpecBase(name="y", paramtype="numeric")
    idps = InterDependencies_(dependencies={yparam: (xparam,)})

    background_run = new_data_set("background-run")
    background_run.set_interdependencies(idps)
    background_run.mark_started(start_bg_writer=True)

    dataset.set_interdependencies(idps)
    received: list[tuple[VALUE, ...]] = []
    try:
        dataset.subscribe(
            lambda results, length, state: received.extend(results),
            min_wait=0,
            min_count=1,
        )
    finally:
        background_run.mark_completed()
        background_run.conn.close()

    dataset.mark_started()
    dataset.add_results([{"x": 1, "y": 2}])
    dataset.mark_completed()
    assert received == [(1, 2)]


def test_subscriber_length_matches_delivered_results(dataset: "DataSet") -> None:
    """
    The length passed to the callback is the number of results in the dataset
    up to and including the ones passed along with it, however the inserts
    and the callbacks interleave.
    """
    xparam = ParamSpecBase(name="x", paramtype="numeric")
    yparam = ParamSpecBase(name="y", paramtype="numeric")
    dataset.set_interdependencies(InterDependencies_(dependencies={yparam: (xparam,)}))
    dataset.mark_started()

    calls: list[tuple[int, int]] = []

    def subscriber(
        results: "list[tuple[VALUE, ...]]", length: int, state: None
    ) -> None:
        calls.append((length, len(results)))

    dataset.subscribe(subscriber, min_wait=0, min_count=1)
    n_results = 1000
    for x in range(n_results):
        dataset.add_results([{"x": x, "y": x}])
    dataset.mark_completed()

    delivered = 0
    for length, n_delivered in calls:
        delivered += n_delivered
        assert length == delivered
    assert delivered == n_results


@pytest.mark.usefixtures("working_subscriber_config")
def test_subscription_from_config(
    dataset: "DataSet",
    basic_subscriber: "Callable[[list[tuple[VALUE]], int, dict | None], None]",
) -> None:
    """
    This test is similar to `test_basic_subscription`, with the only
    difference that another subscriber from a config file is added.
    """
    assert "test_subscriber" in qcodes.config.subscription.subscribers

    xparam = ParamSpecBase(name="x", paramtype="numeric", label="x parameter", unit="V")
    yparam = ParamSpecBase(
        name="y", paramtype="numeric", label="y parameter", unit="Hz"
    )
    idps = InterDependencies_(dependencies={yparam: (xparam,)})
    dataset.set_interdependencies(idps)

    dataset.mark_started()

    sub_id = dataset.subscribe(basic_subscriber, min_wait=0, min_count=1, state={})
    sub_id_c = dataset.subscribe_from_config("test_subscriber")
    assert len(dataset.subscribers) == 2
    assert list(dataset.subscribers.keys()) == [sub_id, sub_id_c]

    expected_state = {}

    # Here we are only testing 2 to reduce the CI time
    for x in range(2):
        y = -(x**2)
        dataset.add_results([{"x": x, "y": y}])
        expected_state[x + 1] = [(x, y)]

        @retry_until_does_not_throw(exception_class_to_expect=AssertionError, tries=10)
        def assert_expected_state() -> "None":
            assert dataset.subscribers[sub_id].state == expected_state
            assert dataset.subscribers[sub_id_c].state == expected_state

        assert_expected_state()


@pytest.mark.usefixtures("broken_subscriber_config")
def test_subscription_from_config_wrong_name(dataset: "DataSet") -> None:
    """
    This test checks that an exception is thrown if a wrong name for a
    subscriber is passed
    """
    assert "test_subscriber" not in qcodes.config.subscription.subscribers
    with pytest.raises(RuntimeError):
        dataset.subscribe_from_config("test_subscriber")
