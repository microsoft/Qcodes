import os
import tempfile

import pytest
import numpy as np

import qcodes as qc
from qcodes.dataset.measurements import DataSaver, Measurement
from qcodes.dataset.param_spec import ParamSpec
from qcodes.dataset.database import initialise_or_create_database_at
from qcodes.dataset.data_export import load_by_id


@pytest.fixture(scope="function")
def empty_temp_db():
    # create a temp database for testing
    with tempfile.TemporaryDirectory() as tmpdirname:
        qc.config["core"]["db_debug"] = False
        initialise_or_create_database_at(os.path.join(tmpdirname, 'tmp.db'))
        yield


@pytest.fixture(scope='function')
def experiment(empty_temp_db):
    e = qc.new_experiment("string-values", sample_name="whatever")
    yield e
    e.conn.close()


def test_string_via_dataset(experiment):
    """
    Test that we can save text into database via DataSet API
    """
    p = ParamSpec("p", "text")

    test_set = qc.new_data_set("test-dataset")
    test_set.add_parameter(p)

    test_set.add_result({"p": "some text"})

    test_set.mark_complete()

    assert test_set.get_data("p") == [["some text"]]


def test_string_via_datasaver(experiment):
    """
    Test that we can save text into database via DataSaver API
    """
    p = ParamSpec("p", "text")

    test_set = qc.new_data_set("test-dataset")
    test_set.add_parameter(p)

    data_saver = DataSaver(
        dataset=test_set, write_period=0, parameters={"p": p})

    data_saver.add_result(("p", "some text"))
    data_saver.flush_data_to_database()

    assert test_set.get_data("p") == [["some text"]]


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

    assert test_set.get_data("p") == [["some text"]]


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
        msg = "It is not possible to save a string value for parameter 'p' " \
              "because its type class is 'numeric', not 'text'."
        with pytest.raises(ValueError, match=msg):
            datasaver.add_result((p, "some text"))


def test_string_with_wrong_paramtype_via_datasaver(experiment):
    """
    Test that it is not possible to add a string value for a non-text
    parameter via DataSaver object
    """
    p = ParamSpec("p", "numeric")

    test_set = qc.new_data_set("test-dataset")
    test_set.add_parameter(p)

    data_saver = DataSaver(
        dataset=test_set, write_period=0, parameters={"p": p})

    try:
        msg = "It is not possible to save a string value for parameter 'p' " \
              "because its type class is 'numeric', not 'text'."
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
    p = ParamSpec("p", "numeric")

    test_set = qc.new_data_set("test-dataset")
    test_set.add_parameter(p)

    test_set.add_result({"p": 'some text'})

    test_set.mark_complete()

    try:
        assert [['some text']] == test_set.get_data("p")
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

    try:
        assert [[item] for item in list_of_strings] == test_set.get_data("p")
    finally:
        test_set.conn.close()
