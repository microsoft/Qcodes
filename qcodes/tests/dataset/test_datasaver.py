import pytest
import numpy as np
from hypothesis import given, strategies as hst

import qcodes as qc
from qcodes.dataset.measurements import DataSaver
from qcodes.dataset.param_spec import ParamSpec
# pylint: disable=unused-import
from qcodes.tests.dataset.temporary_databases import empty_temp_db, experiment

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
def test_default_callback():
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
        test_set.add_metadata('snapshot', 123)
        DataSaver(dataset=test_set, write_period=0, parameters={})
        test_set.mark_complete()
        assert CALLBACK_SNAPSHOT == 123
        assert CALLBACK_RUN_ID > 0
        assert CALLBACK_COUNT > 0
    finally:
        DataSaver.default_callback = None
        if test_set is not None:
            test_set.conn.close()


@pytest.mark.usefixtures("experiment")
def test_numpy_types():
    """
    Test that we can save numpy types in the data set
    """

    p = ParamSpec("p", "numeric")
    test_set = qc.new_data_set("test-dataset")
    test_set.add_parameter(p)

    data_saver = DataSaver(
        dataset=test_set, write_period=0, parameters={"p": p})

    dtypes = [np.int8, np.int16, np.int32, np.int64, np.float16, np.float32,
              np.float64]

    for dtype in dtypes:
        data_saver.add_result(("p", dtype(2)))

    data_saver.flush_data_to_database()
    data = test_set.get_data("p")
    assert data == [[2] for _ in range(len(dtypes))]


@pytest.mark.usefixtures("experiment")
@given(numeric_type=hst.sampled_from([int, float, np.int8, np.int16, np.int32,
                                      np.int64, np.float16, np.float32,
                                      np.float64]))
def test_saving_numeric_values_as_text(numeric_type):
    """
    Test the saving numeric values into 'text' parameter raises an exception
    """
    p = ParamSpec("p", "text")

    test_set = qc.new_data_set("test-dataset")
    test_set.add_parameter(p)

    data_saver = DataSaver(
        dataset=test_set, write_period=0, parameters={"p": p})

    try:
        msg = f"It is not possible to save a numeric value for parameter " \
              f"'{p.name}' because its type class is 'text', not 'numeric'."
        with pytest.raises(ValueError, match=msg):
            data_saver.add_result((p.name, numeric_type(2)))
    finally:
        data_saver.dataset.conn.close()
