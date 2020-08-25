from typing import List, Tuple, Dict, Union

import pytest

import numpy as np
from numpy.testing import assert_allclose

from qcodes import new_data_set
from qcodes.dataset.measurements import Measurement
from qcodes.dataset.sqlite.connection import atomic_transaction
from qcodes.dataset.descriptions.param_spec import ParamSpecBase
from qcodes.dataset.descriptions.dependencies import InterDependencies_
from qcodes.tests.common import retry_until_does_not_throw

VALUE = Union[str, float, List, np.ndarray, bool]


@pytest.mark.usefixtures("experiment")
@pytest.mark.parametrize("bg_writing", [True, False])
def test_nested_measurement_basic(DAC, DMM, bg_writing):
    meas1 = Measurement()
    meas1.register_parameter(DAC.ch1)
    meas1.register_parameter(DMM.v1, setpoints=(DAC.ch1,))

    meas2 = Measurement()
    meas2.register_parameter(DAC.ch2)
    meas2.register_parameter(DMM.v2, setpoints=(DAC.ch2,))

    with meas1.run(write_in_background=bg_writing) as ds1, meas2.run(write_in_background=bg_writing) as ds2:
        for i in range(10):
            DAC.ch1.set(i)
            DAC.ch2.set(i)
            ds1.add_result((DAC.ch1, i),
                           (DMM.v1, DMM.v1()))
            ds2.add_result((DAC.ch2, i),
                           (DMM.v2, DMM.v2()))

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


@pytest.mark.serial
def test_basic_subscription(experiment, basic_subscriber):
    xparam = ParamSpecBase(name='x',
                           paramtype='numeric',
                           label='x parameter',
                           unit='V')
    yparam1 = ParamSpecBase(name='y1',
                            paramtype='numeric',
                            label='y parameter',
                            unit='Hz')
    yparam2 = ParamSpecBase(name='y2',
                            paramtype='numeric',
                            label='y parameter',
                            unit='Hz')

    dataset1 = new_data_set("test-dataset-1")
    idps_1 = InterDependencies_(dependencies={yparam1: (xparam,)})
    dataset1.set_interdependencies(idps_1)
    dataset1.mark_started()

    sub_id_1 = dataset1.subscribe(basic_subscriber, min_wait=0, min_count=1,
                                  state={})

    assert len(dataset1.subscribers) == 1
    assert list(dataset1.subscribers.keys()) == [sub_id_1]

    dataset2 = new_data_set("test-dataset-2")
    idps_2 = InterDependencies_(dependencies={yparam2: (xparam,)})
    dataset2.set_interdependencies(idps_2)
    dataset2.mark_started()

    sub_id_2 = dataset2.subscribe(basic_subscriber, min_wait=0, min_count=1,
                                  state={})

    assert len(dataset2.subscribers) == 1
    assert list(dataset2.subscribers.keys()) == [sub_id_2]

    assert sub_id_1 != sub_id_2


    expected_state_1 = {}
    expected_state_2 = {}

    for x in range(10):
        y1 = -x**2
        y2 = x ** 2
        dataset1.add_results([{'x': x, 'y1': y1}])
        dataset2.add_results([{'x': x, 'y2': y2}])
        expected_state_1[x+1] = [(x, y1)]
        expected_state_2[x + 1] = [(x, y2)]

        @retry_until_does_not_throw(
            exception_class_to_expect=AssertionError, delay=0, tries=10)
        def assert_expected_state():
            assert dataset1.subscribers[sub_id_1].state == expected_state_1
            assert dataset2.subscribers[sub_id_2].state == expected_state_2

        assert_expected_state()

    dataset1.unsubscribe(sub_id_1)
    dataset2.unsubscribe(sub_id_2)

    assert len(dataset2.subscribers) == 0
    assert list(dataset2.subscribers.keys()) == []

    assert len(dataset2.subscribers) == 0
    assert list(dataset2.subscribers.keys()) == []

    # Ensure the trigger for the subscriber has been removed from the database
    get_triggers_sql = "SELECT * FROM sqlite_master WHERE TYPE = 'trigger';"
    triggers1 = atomic_transaction(
        dataset1.conn, get_triggers_sql).fetchall()
    assert len(triggers1) == 0

    triggers2 = atomic_transaction(
        dataset2.conn, get_triggers_sql).fetchall()
    assert len(triggers2) == 0

