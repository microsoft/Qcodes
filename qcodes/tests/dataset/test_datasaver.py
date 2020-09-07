import re

import pytest
import numpy as np
from hypothesis import given, strategies as hst

import qcodes as qc
from qcodes.dataset.measurements import DataSaver
from qcodes.dataset.descriptions.param_spec import ParamSpecBase
from qcodes.dataset.descriptions.dependencies import InterDependencies_

CALLBACK_COUNT = 0
CALLBACK_RUN_ID = None
CALLBACK_SNAPSHOT = None


def callback(result_list, data_set_len, state, run_id, snapshot):
    """
    default_callback example function implemented in the Web UI.
    """
    global CALLBACK_COUNT, CALLBACK_RUN_ID, CALLBACK_SNAPSHOT
    CALLBACK_COUNT += 1
    CALLBACK_RUN_ID = run_id
    CALLBACK_SNAPSHOT = snapshot


def reset_callback_globals():
    global CALLBACK_COUNT, CALLBACK_RUN_ID, CALLBACK_SNAPSHOT
    CALLBACK_COUNT = 0
    CALLBACK_RUN_ID = None
    CALLBACK_SNAPSHOT = None


@pytest.mark.usefixtures("experiment")
@pytest.mark.parametrize("bg_writing", [True, False])
def test_default_callback(bg_writing):
    """
    The Web UI needs to know the results of an experiment with the metadata.
    So a default_callback class variable is set by the Web UI with a callback
    to introspect the data.
    """
    test_set = None
    reset_callback_globals()

    try:
        DataSaver.default_callback = {
            'run_tables_subscription_callback': callback,
            'run_tables_subscription_min_wait': 1,
            'run_tables_subscription_min_count': 2}

        test_set = qc.new_data_set("test-dataset")
        test_set.add_metadata('snapshot', 'reasonable_snapshot')
        DataSaver(dataset=test_set,
                  write_period=0,
                  interdeps=InterDependencies_)
        test_set.mark_started(start_bg_writer=bg_writing)
        test_set.mark_completed()
        assert CALLBACK_SNAPSHOT == 'reasonable_snapshot'
        assert CALLBACK_RUN_ID > 0
        assert CALLBACK_COUNT > 0
    finally:
        DataSaver.default_callback = None
        if test_set is not None:
            test_set.conn.close()


@pytest.mark.usefixtures("experiment")
@pytest.mark.parametrize("bg_writing", [True, False])
def test_numpy_types(bg_writing):
    """
    Test that we can save numpy types in the data set
    """

    p = ParamSpecBase(name="p", paramtype="numeric")
    test_set = qc.new_data_set("test-dataset")
    test_set.set_interdependencies(InterDependencies_(standalones=(p,)))
    test_set.mark_started(start_bg_writer=bg_writing)

    idps = InterDependencies_(standalones=(p,))

    data_saver = DataSaver(
        dataset=test_set, write_period=0, interdeps=idps)

    dtypes = [np.int8, np.int16, np.int32, np.int64, np.float16, np.float32,
              np.float64]

    for dtype in dtypes:
        data_saver.add_result(("p", dtype(2)))

    data_saver.flush_data_to_database()
    test_set.mark_completed()
    data = test_set.get_parameter_data("p")["p"]["p"]
    expected_data = np.ones(len(dtypes))
    expected_data[:] = 2
    np.testing.assert_array_equal(data, expected_data)


@pytest.mark.usefixtures("experiment")
@pytest.mark.parametrize('numeric_type',
                         [int, float, np.int8, np.int16, np.int32, np.int64,
                          np.float16, np.float32, np.float64])
@pytest.mark.parametrize("bg_writing", [True, False])
def test_saving_numeric_values_as_text(numeric_type, bg_writing):
    """
    Test the saving numeric values into 'text' parameter raises an exception
    """
    p = ParamSpecBase("p", "text")

    test_set = qc.new_data_set("test-dataset")
    test_set.set_interdependencies(InterDependencies_(standalones=(p,)))
    test_set.mark_started(start_bg_writer=bg_writing)

    idps = InterDependencies_(standalones=(p,))

    data_saver = DataSaver(
        dataset=test_set, write_period=0, interdeps=idps)

    try:
        value = numeric_type(2)

        gottype = np.array(value).dtype

        msg = re.escape(f'Parameter {p.name} is of type '
                        f'"{p.type}", but got a result of '
                        f'type {gottype} ({value}).')
        with pytest.raises(ValueError, match=msg):
            data_saver.add_result((p.name, value))
    finally:
        data_saver.dataset.mark_completed()
        data_saver.dataset.conn.close()
