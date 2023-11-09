from typing import Union

import hypothesis.strategies as hst
import numpy as np
import pytest
from hypothesis import given, settings
from numpy.testing import assert_allclose

from qcodes.dataset import Measurement, new_data_set
from qcodes.dataset.descriptions.dependencies import InterDependencies_
from qcodes.dataset.descriptions.param_spec import ParamSpecBase
from qcodes.dataset.sqlite.connection import atomic_transaction
from tests.common import retry_until_does_not_throw

VALUE = Union[str, float, list, np.ndarray, bool]


@pytest.mark.usefixtures("experiment")
@pytest.mark.parametrize("bg_writing", [True, False])
def test_nested_measurement_basic(DAC, DMM, bg_writing) -> None:
    meas1 = Measurement()
    meas1.register_parameter(DAC.ch1)
    meas1.register_parameter(DMM.v1, setpoints=(DAC.ch1,))

    meas2 = Measurement()
    meas2.register_parameter(DAC.ch2)
    meas2.register_parameter(DMM.v2, setpoints=(DAC.ch2,))

    with meas1.run(write_in_background=bg_writing) as ds1, meas2.run(
        write_in_background=bg_writing
    ) as ds2:
        for i in range(10):
            DAC.ch1.set(i)
            DAC.ch2.set(i)
            ds1.add_result((DAC.ch1, i), (DMM.v1, DMM.v1()))
            ds2.add_result((DAC.ch2, i), (DMM.v2, DMM.v2()))

    data1 = ds1.dataset.get_parameter_data()["dummy_dmm_v1"]
    assert len(data1.keys()) == 2
    assert "dummy_dmm_v1" in data1.keys()
    assert "dummy_dac_ch1" in data1.keys()
    assert_allclose(data1["dummy_dmm_v1"], np.zeros(10))
    assert_allclose(data1["dummy_dac_ch1"], np.arange(10))

    data2 = ds2.dataset.get_parameter_data()["dummy_dmm_v2"]
    assert len(data2.keys()) == 2
    assert "dummy_dmm_v2" in data2.keys()
    assert "dummy_dac_ch2" in data2.keys()
    assert_allclose(data2["dummy_dmm_v2"], np.zeros(10))
    assert_allclose(data2["dummy_dac_ch2"], np.arange(10))


@pytest.mark.usefixtures("experiment")
@pytest.mark.parametrize("bg_writing", [True, False])
def test_nested_measurement(bg_writing) -> None:
    meas1 = Measurement()
    meas1.register_custom_parameter("foo1")
    meas1.register_custom_parameter("bar1", setpoints=("foo1",))

    meas2 = Measurement()
    meas2.register_custom_parameter("foo2")
    meas2.register_custom_parameter("bar2", setpoints=("foo2",))

    with meas1.run(write_in_background=bg_writing) as ds1, meas2.run(
        write_in_background=bg_writing
    ) as ds2:
        for i in range(10):
            ds1.add_result(("foo1", i), ("bar1", i**2))
            ds2.add_result(("foo2", 2 * i), ("bar2", (2 * i) ** 2))

    data1 = ds1.dataset.get_parameter_data()["bar1"]
    assert len(data1.keys()) == 2
    assert "foo1" in data1.keys()
    assert "bar1" in data1.keys()

    assert_allclose(data1["foo1"], np.arange(10))
    assert_allclose(data1["bar1"], np.arange(10) ** 2)

    data2 = ds2.dataset.get_parameter_data()["bar2"]
    assert len(data2.keys()) == 2
    assert "foo2" in data2.keys()
    assert "bar2" in data2.keys()
    assert_allclose(data2["foo2"], np.arange(0, 20, 2))
    assert_allclose(data2["bar2"], np.arange(0, 20, 2) ** 2)


@pytest.mark.usefixtures("experiment")
@pytest.mark.parametrize("bg_writing", [True, False])
@settings(deadline=None, max_examples=25)
@given(
    outer_len=hst.integers(min_value=1, max_value=100),
    inner_len1=hst.integers(min_value=1, max_value=1000),
    inner_len2=hst.integers(min_value=1, max_value=1000),
)
def test_nested_measurement_array(
    bg_writing, outer_len, inner_len1, inner_len2
) -> None:
    meas1 = Measurement()
    meas1.register_custom_parameter("foo1", paramtype="numeric")
    meas1.register_custom_parameter("bar1spt", paramtype="array")
    meas1.register_custom_parameter(
        "bar1", setpoints=("foo1", "bar1spt"), paramtype="array"
    )

    meas2 = Measurement()
    meas2.register_custom_parameter("foo2", paramtype="numeric")
    meas2.register_custom_parameter("bar2spt", paramtype="array")
    meas2.register_custom_parameter(
        "bar2",
        setpoints=(
            "foo2",
            "bar2spt",
        ),
        paramtype="array",
    )

    with meas1.run(write_in_background=bg_writing) as ds1, meas2.run(
        write_in_background=bg_writing
    ) as ds2:
        for i in range(outer_len):
            bar1sptdata = np.arange(inner_len1)
            bar2sptdata = np.arange(inner_len2)
            ds1.add_result(
                ("foo1", i),
                ("bar1spt", bar1sptdata),
                ("bar1", np.ones(inner_len1) * i * bar1sptdata),
            )
            ds2.add_result(
                ("foo2", i),
                ("bar2spt", bar2sptdata),
                ("bar2", np.ones(inner_len2) * i * bar2sptdata),
            )

    data1 = ds1.dataset.get_parameter_data()["bar1"]
    assert len(data1.keys()) == 3
    assert "foo1" in data1.keys()
    assert "bar1spt" in data1.keys()
    assert "bar1" in data1.keys()

    expected_foo1_data = np.repeat(np.arange(outer_len), inner_len1).reshape(
        outer_len, inner_len1
    )
    expected_bar1spt_data = np.tile(np.arange(inner_len1), (outer_len, 1))

    assert_allclose(data1["foo1"], expected_foo1_data)
    assert_allclose(data1["bar1spt"], expected_bar1spt_data)
    assert_allclose(data1["bar1"], expected_foo1_data * expected_bar1spt_data)

    data2 = ds2.dataset.get_parameter_data()["bar2"]
    assert len(data2.keys()) == 3
    assert "foo2" in data2.keys()
    assert "bar2spt" in data2.keys()
    assert "bar2" in data2.keys()

    expected_foo2_data = np.repeat(np.arange(outer_len), inner_len2).reshape(
        outer_len, inner_len2
    )
    expected_bar2spt_data = np.tile(np.arange(inner_len2), (outer_len, 1))

    assert_allclose(data2["foo2"], expected_foo2_data)
    assert_allclose(data2["bar2spt"], expected_bar2spt_data)
    assert_allclose(data2["bar2"], expected_foo2_data * expected_bar2spt_data)


@pytest.fixture(scope="function")
def basic_subscriber():
    """
    A basic subscriber that just puts results and length into
    state
    """

    def subscriber(results: list[tuple[VALUE]], length: int, state: dict) -> None:
        state[length] = results

    return subscriber


@pytest.mark.flaky(reruns=5)
@pytest.mark.serial
def test_subscription_on_dual_datasets(experiment, basic_subscriber) -> None:
    xparam = ParamSpecBase(name="x", paramtype="numeric", label="x parameter", unit="V")
    yparam1 = ParamSpecBase(
        name="y1", paramtype="numeric", label="y parameter", unit="Hz"
    )
    yparam2 = ParamSpecBase(
        name="y2", paramtype="numeric", label="y parameter", unit="Hz"
    )

    dataset1 = new_data_set("test-dataset-1")
    idps_1 = InterDependencies_(dependencies={yparam1: (xparam,)})
    dataset1.set_interdependencies(idps_1)
    dataset1.mark_started()

    sub_id_1 = dataset1.subscribe(basic_subscriber, min_wait=0, min_count=1, state={})

    assert len(dataset1.subscribers) == 1
    assert list(dataset1.subscribers.keys()) == [sub_id_1]

    dataset2 = new_data_set("test-dataset-2")
    idps_2 = InterDependencies_(dependencies={yparam2: (xparam,)})
    dataset2.set_interdependencies(idps_2)
    dataset2.mark_started()

    sub_id_2 = dataset2.subscribe(basic_subscriber, min_wait=0, min_count=1, state={})

    assert len(dataset2.subscribers) == 1
    assert list(dataset2.subscribers.keys()) == [sub_id_2]

    assert sub_id_1 != sub_id_2

    expected_state_1 = {}
    expected_state_2 = {}

    for x in range(10):
        y1 = -(x**2)
        y2 = x**2
        dataset1.add_results([{"x": x, "y1": y1}])
        dataset2.add_results([{"x": x, "y2": y2}])
        expected_state_1[x + 1] = [(x, y1)]
        expected_state_2[x + 1] = [(x, y2)]

        @retry_until_does_not_throw(
            exception_class_to_expect=AssertionError, delay=0.5, tries=10
        )
        def assert_expected_state():
            assert dataset1.subscribers[sub_id_1].state == expected_state_1
            assert dataset2.subscribers[sub_id_2].state == expected_state_2

        assert_expected_state()

    dataset1.unsubscribe(sub_id_1)
    dataset2.unsubscribe(sub_id_2)

    assert len(dataset1.subscribers) == 0
    assert list(dataset1.subscribers.keys()) == []

    assert len(dataset2.subscribers) == 0
    assert list(dataset2.subscribers.keys()) == []

    # Ensure the trigger for the subscriber has been removed from the database
    get_triggers_sql = "SELECT * FROM sqlite_master WHERE TYPE = 'trigger';"
    triggers1 = atomic_transaction(dataset1.conn, get_triggers_sql).fetchall()
    assert len(triggers1) == 0

    triggers2 = atomic_transaction(dataset2.conn, get_triggers_sql).fetchall()
    assert len(triggers2) == 0
