import pytest
import os
import tempfile
import numpy as np

import qcodes as qc
from qcodes.dataset.measurements import DataSaver, Measurement
from qcodes.dataset.param_spec import ParamSpec
from qcodes.dataset.database import initialise_or_create_database_at
from qcodes.dataset.data_export import load_by_id


# These fixture can't be imported from test_dataset_basic because
# Codacy/PR Quality Review will think they are unused.
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


def test_string_without_specifying_paramtype(experiment):
    """
    Test that an exception occurs when loading the string data if when
    registering a string parameter the paramtype was not explicitly set to
    'text'
    """
    p = qc.Parameter('p', label='String parameter', unit='', get_cmd=None,
                     set_cmd=None, initial_value='some text')

    meas = Measurement(experiment)
    meas.register_parameter(p)  # intentionally forgot `paramtype='text'`

    with meas.run() as datasaver:
        datasaver.add_result((p, "some text"))

    test_set = load_by_id(datasaver.run_id)

    try:
        with pytest.raises(ValueError,
                           match="could not convert string to float: b'some "
                                 "text'"):
            assert test_set.get_data("p") == [["some text"]]
    finally:
        test_set.conn.close()
