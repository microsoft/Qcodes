import re

import hypothesis.strategies as hst
from hypothesis import given
import hypothesis.extra.numpy as hypnumpy
import pytest
import numpy as np

import qcodes as qc
from qcodes.dataset.measurements import DataSaver, Measurement
from qcodes.dataset.descriptions.dependencies import InterDependencies_
from qcodes.dataset.descriptions.param_spec import ParamSpecBase
from qcodes.dataset.data_export import load_by_id


def test_string_via_dataset(experiment):
    """
    Test that we can save text into database via DataSet API
    """
    p = ParamSpecBase("p", "text")

    test_set = qc.new_data_set("test-dataset")
    idps = InterDependencies_(standalones=(p,))
    test_set.set_interdependencies(idps)
    test_set.mark_started()

    test_set.add_results([{"p": "some text"}])

    test_set.mark_completed()

    assert test_set.get_parameter_data()["p"]["p"] == [["some text"]]


def test_string_via_datasaver(experiment):
    """
    Test that we can save text into database via DataSaver API
    """
    p = ParamSpecBase(name="p", paramtype="text")

    test_set = qc.new_data_set("test-dataset")
    idps = InterDependencies_(standalones=(p,))
    test_set.set_interdependencies(idps)
    test_set.mark_started()

    idps = InterDependencies_(standalones=(p,))

    data_saver = DataSaver(
        dataset=test_set, write_period=0, interdeps=idps)

    data_saver.add_result(("p", "some text"))
    data_saver.flush_data_to_database()

    assert test_set.get_parameter_data()["p"]["p"] == np.array(["some text"])


def test_string(experiment):
    """
    Test that we can save text into database via Measurement API
    """
    p = qc.Parameter('p', label='String parameter', unit='', get_cmd=None,
                     set_cmd=None, initial_value='some text')

    meas = Measurement(experiment)
    meas.register_parameter(p, paramtype='text')

    with meas.run() as datasaver:
        datasaver.add_result((p, "some text"))

    test_set = load_by_id(datasaver.run_id)

    assert test_set.get_parameter_data()["p"]["p"] == np.array(["some text"])


def test_string_with_wrong_paramtype(experiment):
    """
    Test that an exception occurs when saving string data if when registering a
    string parameter the paramtype was not set to 'text'
    """
    p = qc.Parameter('p', label='String parameter', unit='', get_cmd=None,
                     set_cmd=None, initial_value='some text')

    meas = Measurement(experiment)
    # intentionally forget `paramtype='text'`, so that the default 'numeric'
    # is used, and an exception is raised later
    meas.register_parameter(p)

    with meas.run() as datasaver:
        msg = re.escape('Parameter p is of type "numeric", but got a '
                        "result of type <U9 (some text).")
        with pytest.raises(ValueError, match=msg):
            datasaver.add_result((p, "some text"))


def test_string_with_wrong_paramtype_via_datasaver(experiment):
    """
    Test that it is not possible to add a string value for a non-text
    parameter via DataSaver object
    """
    p = ParamSpecBase("p", "numeric")

    test_set = qc.new_data_set("test-dataset")
    idps = InterDependencies_(standalones=(p,))
    test_set.set_interdependencies(idps)
    test_set.mark_started()

    idps = InterDependencies_(standalones=(p,))

    data_saver = DataSaver(
        dataset=test_set, write_period=0, interdeps=idps)

    try:
        msg = re.escape('Parameter p is of type "numeric", but got a '
                        "result of type <U9 (some text).")
        with pytest.raises(ValueError, match=msg):
            data_saver.add_result(("p", "some text"))
    finally:
        data_saver.dataset.conn.close()


def test_string_saved_and_loaded_as_numeric_via_dataset(experiment):
    """
    Test that it is possible to save a string value of a non-'text' parameter
    via DataSet API, and, importantly, to retrieve it thanks to the
    flexibility of `_convert_numeric` converter function.
    """
    p = ParamSpecBase("p", "numeric")

    test_set = qc.new_data_set("test-dataset")
    idps = InterDependencies_(standalones=(p,))
    test_set.set_interdependencies(idps)
    test_set.mark_started()

    test_set.add_results([{"p": 'some text'}])

    test_set.mark_completed()

    try:
        assert np.array(['some text']) == test_set.get_parameter_data()["p"]["p"]
    finally:
        test_set.conn.close()


def test_list_of_strings(experiment):
    """
    Test saving list of strings via DataSaver
    """
    p_values = ["X_Y", "X_X", "X_I", "I_I"]
    list_of_strings = list(np.random.choice(p_values, (10,)))

    p = qc.Parameter('p', label='String parameter', unit='', get_cmd=None,
                     set_cmd=None, initial_value='X_Y')

    meas = Measurement(experiment)
    meas.register_parameter(p, paramtype='text')

    with meas.run() as datasaver:
        datasaver.add_result((p, list_of_strings))

    test_set = load_by_id(datasaver.run_id)
    expec_data = np.array([item for item in list_of_strings])
    actual_data = test_set.get_parameter_data()["p"]["p"]

    try:
        np.testing.assert_array_equal(actual_data, expec_data)
    finally:
        test_set.conn.close()


@given(
    p_values=hypnumpy.arrays(
        dtype=hst.sampled_from(
            (hypnumpy.unicode_string_dtypes(),
             hypnumpy.byte_string_dtypes(),
             hypnumpy.timedelta64_dtypes(),
             hypnumpy.datetime64_dtypes())
        ),
        shape=hypnumpy.array_shapes()
    )
)
def test_string_and_date_data_in_array(experiment, p_values):
    p = qc.Parameter('p', label='String parameter', unit='', get_cmd=None,
                     set_cmd=None, initial_value=p_values)

    meas = Measurement(experiment)
    meas.register_parameter(p, paramtype='array')

    with meas.run() as datasaver:
        datasaver.add_result((p, p.get()))
    actual_data = datasaver.dataset.get_parameter_data()["p"]["p"]
    np.testing.assert_array_equal(
        actual_data,
        p_values.reshape((1, *p_values.shape))
    )
