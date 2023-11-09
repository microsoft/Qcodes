import os
import random
import re
import uuid
from contextlib import contextmanager
from os.path import getmtime
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import qcodes as qc
import tests.dataset
from qcodes.dataset import do1d, do2d
from qcodes.dataset.data_set import (
    DataSet,
    generate_dataset_table,
    load_by_counter,
    load_by_guid,
    load_by_id,
    load_by_run_spec,
)
from qcodes.dataset.database_extract_runs import extract_runs_into_db
from qcodes.dataset.experiment_container import (
    Experiment,
    load_experiment_by_name,
    load_or_create_experiment,
)
from qcodes.dataset.linked_datasets.links import Link
from qcodes.dataset.measurements import Measurement
from qcodes.dataset.sqlite.connection import path_to_dbfile
from qcodes.dataset.sqlite.database import get_db_version_and_newest_available_version
from qcodes.dataset.sqlite.queries import get_experiments
from qcodes.instrument_drivers.mock_instruments import DummyInstrument
from qcodes.station import Station
from tests.common import error_caused_by, skip_if_no_fixtures


@contextmanager
def raise_if_file_changed(path_to_file: str):
    """
    Context manager that raises if a file is modified.
    On Windows, the OS modification time resolution is 100 ns
    """
    pre_operation_time = getmtime(path_to_file)
    # we don't want to catch and re-raise anything, since there is no clean-up
    # that we need to perform. Hence no try-except here
    yield
    post_operation_time = getmtime(path_to_file)
    if pre_operation_time != post_operation_time:
        raise RuntimeError(f"File {path_to_file} was modified.")


@pytest.fixture(scope="function", name="inst")
def _make_inst():
    """
    Dummy instrument for testing, ensuring that it's instance gets closed
    and removed from the global register of instruments, which, if not done,
    make break other tests
    """
    inst = DummyInstrument("extract_run_inst", gates=["back", "plunger", "cutter"])
    yield inst
    inst.close()


def test_missing_runs_raises(two_empty_temp_db_connections, some_interdeps) -> None:
    """
    Test that an error is raised if we attempt to extract a run not present in
    the source DB
    """
    source_conn, target_conn = two_empty_temp_db_connections

    source_exp_1 = Experiment(conn=source_conn)

    # make 5 runs in first experiment

    exp_1_run_ids = []
    for _ in range(5):
        source_dataset = DataSet(conn=source_conn, exp_id=source_exp_1.exp_id)
        exp_1_run_ids.append(source_dataset.run_id)
        source_dataset.set_interdependencies(some_interdeps[1])

        source_dataset.mark_started()

        for val in range(10):
            source_dataset.add_results(
                [{name: val for name in some_interdeps[1].names}]
            )
        source_dataset.mark_completed()

    source_path = path_to_dbfile(source_conn)
    target_path = path_to_dbfile(target_conn)

    run_ids = [1, 8, 5, 3, 2, 4, 4, 4, 7, 8]
    wrong_ids = [8, 7, 8]

    expected_err = (
        "Error: not all run_ids exist in the source database. "
        "The following run(s) is/are not present: "
        f"{wrong_ids}"
    )

    with pytest.raises(ValueError, match=re.escape(expected_err)):
        extract_runs_into_db(source_path, target_path, *run_ids)


def test_basic_extraction(two_empty_temp_db_connections, some_interdeps) -> None:
    source_conn, target_conn = two_empty_temp_db_connections

    source_path = path_to_dbfile(source_conn)
    target_path = path_to_dbfile(target_conn)

    type_casters: dict[str, Callable] = {
        "numeric": float,
        "array": (lambda x: np.array(x) if hasattr(x, "__iter__") else np.array([x])),
        "text": str,
    }

    source_exp = Experiment(conn=source_conn)
    source_dataset = DataSet(conn=source_conn, name="basic_copy_paste_name")

    with pytest.raises(RuntimeError) as excinfo:
        extract_runs_into_db(source_path, target_path, source_dataset.run_id)

    assert error_caused_by(
        excinfo,
        (
            "Dataset not completed. An incomplete "
            "dataset can not be copied. The "
            "incomplete dataset has GUID: "
            f"{source_dataset.guid} and run_id: "
            f"{source_dataset.run_id}"
        ),
    )

    source_dataset.set_interdependencies(some_interdeps[0])

    source_dataset.parent_dataset_links = [
        Link(head=source_dataset.guid, tail=str(uuid.uuid4()), edge_type="test_link")
    ]
    source_dataset.mark_started()

    for value in range(10):
        result = {
            ps.name: type_casters[ps.type](value) for ps in some_interdeps[0].paramspecs
        }
        source_dataset.add_results([result])

    source_dataset.add_metadata("goodness", "fair")
    source_dataset.add_metadata("test", True)

    source_dataset.mark_completed()

    assert source_dataset.run_id == source_dataset.captured_run_id

    extract_runs_into_db(source_path, target_path, source_dataset.run_id)

    target_exp = Experiment(conn=target_conn, exp_id=1)

    length1 = len(target_exp)
    assert length1 == 1

    # trying to insert the same run again should be a NOOP
    with raise_if_file_changed(target_path):
        extract_runs_into_db(source_path, target_path, source_dataset.run_id)

    assert len(target_exp) == length1

    target_dataset = DataSet(conn=target_conn, run_id=1)

    # Now make the interesting comparisons: are the target objects the same as
    # the source objects?

    assert source_dataset.the_same_dataset_as(target_dataset)
    assert source_dataset.parameters is not None
    assert target_dataset.parameters is not None
    source_data = source_dataset.get_parameter_data(
        *source_dataset.parameters.split(",")
    )
    target_data = target_dataset.get_parameter_data(
        *target_dataset.parameters.split(",")
    )

    for outkey, outval in source_data.items():
        for inkey, inval in outval.items():
            np.testing.assert_array_equal(inval, target_data[outkey][inkey])

    exp_attrs = ["name", "sample_name", "format_string", "started_at", "finished_at"]

    for exp_attr in exp_attrs:
        assert getattr(source_exp, exp_attr) == getattr(target_exp, exp_attr)

    # trying to insert the same run again should be a NOOP
    with raise_if_file_changed(target_path):
        extract_runs_into_db(source_path, target_path, source_dataset.run_id)


def test_real_dataset_1d(two_empty_temp_db_connections, inst) -> None:
    source_conn, target_conn = two_empty_temp_db_connections

    source_path = path_to_dbfile(source_conn)
    target_path = path_to_dbfile(target_conn)

    source_exp = load_or_create_experiment(experiment_name="myexp", conn=source_conn)

    source_dataset, _, _ = do1d(inst.back, 0, 1, 10, 0, inst.plunger, exp=source_exp)

    extract_runs_into_db(source_path, target_path, source_dataset.run_id)

    target_dataset = load_by_guid(source_dataset.guid, conn=target_conn)
    assert isinstance(source_dataset, DataSet)
    assert source_dataset.the_same_dataset_as(target_dataset)
    # explicit regression  test for https://github.com/QCoDeS/Qcodes/issues/3953
    assert source_dataset.description.shapes == {"extract_run_inst_plunger": (10,)}
    assert source_dataset.description.shapes == target_dataset.description.shapes

    source_data = source_dataset.get_parameter_data()["extract_run_inst_plunger"]
    target_data = target_dataset.get_parameter_data()["extract_run_inst_plunger"]

    for source_data_vals, target_data_vals in zip(
        source_data.values(), target_data.values()
    ):
        assert_array_equal(source_data_vals, target_data_vals)


def test_real_dataset_1d_pathlib_path(two_empty_temp_db_connections, inst) -> None:
    source_conn, target_conn = two_empty_temp_db_connections

    source_path = Path(path_to_dbfile(source_conn))
    target_path = Path(path_to_dbfile(target_conn))

    source_exp = load_or_create_experiment(experiment_name="myexp", conn=source_conn)

    source_dataset, _, _ = do1d(inst.back, 0, 1, 10, 0, inst.plunger, exp=source_exp)

    extract_runs_into_db(source_path, target_path, source_dataset.run_id)

    target_dataset = load_by_guid(source_dataset.guid, conn=target_conn)
    assert isinstance(source_dataset, DataSet)
    assert source_dataset.the_same_dataset_as(target_dataset)
    # explicit regression  test for https://github.com/QCoDeS/Qcodes/issues/3953
    assert source_dataset.description.shapes == {"extract_run_inst_plunger": (10,)}
    assert source_dataset.description.shapes == target_dataset.description.shapes

    source_data = source_dataset.get_parameter_data()["extract_run_inst_plunger"]
    target_data = target_dataset.get_parameter_data()["extract_run_inst_plunger"]

    for source_data_vals, target_data_vals in zip(
        source_data.values(), target_data.values()
    ):
        assert_array_equal(source_data_vals, target_data_vals)


def test_real_dataset_2d(two_empty_temp_db_connections, inst) -> None:
    source_conn, target_conn = two_empty_temp_db_connections

    source_path = path_to_dbfile(source_conn)
    target_path = path_to_dbfile(target_conn)

    source_exp = load_or_create_experiment(experiment_name="myexp", conn=source_conn)

    source_dataset, _, _ = do2d(
        inst.back, 0, 1, 10, 0, inst.plunger, 0, 0.1, 15, 0, inst.cutter, exp=source_exp
    )

    extract_runs_into_db(source_path, target_path, source_dataset.run_id)

    target_dataset = load_by_guid(source_dataset.guid, conn=target_conn)
    assert isinstance(source_dataset, DataSet)
    assert source_dataset.the_same_dataset_as(target_dataset)
    # explicit regression  test for https://github.com/QCoDeS/Qcodes/issues/3953
    assert source_dataset.description.shapes == {"extract_run_inst_cutter": (10, 15)}
    assert source_dataset.description.shapes == target_dataset.description.shapes

    source_data = source_dataset.get_parameter_data()["extract_run_inst_cutter"]
    target_data = target_dataset.get_parameter_data()["extract_run_inst_cutter"]

    for source_data_vals, target_data_vals in zip(
        source_data.values(), target_data.values()
    ):
        assert_array_equal(source_data_vals, target_data_vals)


def test_correct_experiment_routing(
    two_empty_temp_db_connections, some_interdeps
) -> None:
    """
    Test that existing experiments are correctly identified AND that multiple
    insertions of the same runs don't matter (run insertion is idempotent)
    """
    source_conn, target_conn = two_empty_temp_db_connections

    source_exp_1 = Experiment(conn=source_conn)

    # make 5 runs in first experiment

    exp_1_run_ids = []
    for _ in range(5):
        source_dataset = DataSet(conn=source_conn, exp_id=source_exp_1.exp_id)
        exp_1_run_ids.append(source_dataset.run_id)

        source_dataset.set_interdependencies(some_interdeps[1])

        source_dataset.mark_started()

        for val in range(10):
            source_dataset.add_results(
                [{name: val for name in some_interdeps[1].names}]
            )
        source_dataset.mark_completed()

    # make a new experiment with 1 run

    source_exp_2 = Experiment(conn=source_conn)
    ds = DataSet(conn=source_conn, exp_id=source_exp_2.exp_id, name="lala")
    exp_2_run_ids = [ds.run_id]

    ds.set_interdependencies(some_interdeps[1])

    ds.mark_started()

    for val in range(10):
        ds.add_results([{name: val for name in some_interdeps[1].names}])

    ds.mark_completed()

    source_path = path_to_dbfile(source_conn)
    target_path = path_to_dbfile(target_conn)

    # now copy 2 runs
    extract_runs_into_db(source_path, target_path, *exp_1_run_ids[:2])

    target_exp1 = Experiment(conn=target_conn, exp_id=1)

    assert len(target_exp1) == 2

    # copy two other runs, one of them already in
    extract_runs_into_db(source_path, target_path, *exp_1_run_ids[1:3])

    assert len(target_exp1) == 3

    # insert run from different experiment
    extract_runs_into_db(source_path, target_path, ds.run_id)

    assert len(target_exp1) == 3

    target_exp2 = Experiment(conn=target_conn, exp_id=2)

    assert len(target_exp2) == 1

    # finally insert every single run from experiment 1

    extract_runs_into_db(source_path, target_path, *exp_1_run_ids)

    # check for idempotency once more by inserting all the runs but in another
    # order
    with raise_if_file_changed(target_path):
        extract_runs_into_db(source_path, target_path, *exp_1_run_ids[::-1])

    target_exps = get_experiments(target_conn)

    assert len(target_exps) == 2
    assert len(target_exp1) == 5
    assert len(target_exp2) == 1

    # check that all the datasets match up
    for run_id in exp_1_run_ids + exp_2_run_ids:
        source_ds = DataSet(conn=source_conn, run_id=run_id)
        target_ds = load_by_guid(guid=source_ds.guid, conn=target_conn)
        assert isinstance(target_ds, DataSet)

        assert source_ds.the_same_dataset_as(target_ds)
        assert source_ds.parameters is not None
        assert target_ds.parameters is not None
        source_data = source_ds.get_parameter_data(*source_ds.parameters.split(","))
        target_data = target_ds.get_parameter_data(*target_ds.parameters.split(","))

        for outkey, outval in source_data.items():
            for inkey, inval in outval.items():
                np.testing.assert_array_equal(inval, target_data[outkey][inkey])


def test_runs_from_different_experiments_raises(
    two_empty_temp_db_connections, some_interdeps
) -> None:
    """
    Test that inserting runs from multiple experiments raises
    """
    source_conn, target_conn = two_empty_temp_db_connections

    source_path = path_to_dbfile(source_conn)
    target_path = path_to_dbfile(target_conn)

    source_exp_1 = Experiment(conn=source_conn)
    source_exp_2 = Experiment(conn=source_conn)

    # make 5 runs in first experiment

    exp_1_run_ids = []
    for _ in range(5):
        source_dataset = DataSet(conn=source_conn, exp_id=source_exp_1.exp_id)
        exp_1_run_ids.append(source_dataset.run_id)

        source_dataset.set_interdependencies(some_interdeps[1])

        source_dataset.mark_started()

        for val in range(10):
            source_dataset.add_results(
                [{name: val for name in some_interdeps[1].names}]
            )
        source_dataset.mark_completed()

    # make 5 runs in second experiment

    exp_2_run_ids = []
    for _ in range(5):
        source_dataset = DataSet(conn=source_conn, exp_id=source_exp_2.exp_id)
        exp_2_run_ids.append(source_dataset.run_id)

        source_dataset.set_interdependencies(some_interdeps[1])

        source_dataset.mark_started()

        for val in range(10):
            source_dataset.add_results(
                [{name: val for name in some_interdeps[1].names}]
            )
        source_dataset.mark_completed()

    run_ids = exp_1_run_ids + exp_2_run_ids
    source_exp_ids = np.unique([1, 2])
    matchstring = (
        "Did not receive runs from a single experiment\\. "
        f"Got runs from experiments {source_exp_ids}"
    )
    # make the matchstring safe to use as a regexp
    matchstring = matchstring.replace("[", "\\[").replace("]", "\\]")
    with pytest.raises(ValueError, match=matchstring):
        extract_runs_into_db(source_path, target_path, *run_ids)


def test_extracting_dataless_run(two_empty_temp_db_connections) -> None:
    """
    Although contrived, it could happen that a run with no data is extracted
    """
    source_conn, target_conn = two_empty_temp_db_connections

    source_path = path_to_dbfile(source_conn)
    target_path = path_to_dbfile(target_conn)

    Experiment(conn=source_conn)

    source_ds = DataSet(conn=source_conn)
    source_ds.mark_started()
    source_ds.mark_completed()

    extract_runs_into_db(source_path, target_path, source_ds.run_id)

    loaded_ds = DataSet(conn=target_conn, run_id=1)

    assert loaded_ds.the_same_dataset_as(source_ds)


def test_result_table_naming_and_run_id(
    two_empty_temp_db_connections, some_interdeps
) -> None:
    """
    Check that a correct result table name is given and that a correct run_id
    is assigned
    """
    source_conn, target_conn = two_empty_temp_db_connections

    source_path = path_to_dbfile(source_conn)
    target_path = path_to_dbfile(target_conn)

    source_exp1 = Experiment(conn=source_conn)
    source_ds_1_1 = DataSet(conn=source_conn, exp_id=source_exp1.exp_id)
    source_ds_1_1.set_interdependencies(some_interdeps[1])

    source_ds_1_1.mark_started()
    source_ds_1_1.add_results([{name: 0.0 for name in some_interdeps[1].names}])
    source_ds_1_1.mark_completed()

    source_exp2 = Experiment(conn=source_conn)
    source_ds_2_1 = DataSet(conn=source_conn, exp_id=source_exp2.exp_id)
    source_ds_2_1.set_interdependencies(some_interdeps[1])
    source_ds_2_1.mark_started()
    source_ds_2_1.add_results([{name: 0.0 for name in some_interdeps[1].names}])
    source_ds_2_1.mark_completed()
    source_ds_2_2 = DataSet(
        conn=source_conn, exp_id=source_exp2.exp_id, name="customname"
    )

    source_ds_2_2.set_interdependencies(some_interdeps[1])
    source_ds_2_2.mark_started()
    source_ds_2_2.add_results([{name: 0.0 for name in some_interdeps[1].names}])
    source_ds_2_2.mark_completed()

    extract_runs_into_db(source_path, target_path, source_ds_2_2.run_id)

    # The target ds ought to have a runs table "results-1-1"
    # and ought to be the same dataset as its "ancestor"
    target_ds = DataSet(conn=target_conn, run_id=1)

    assert target_ds.name == "customname"
    assert target_ds.table_name == "results-1-1"
    assert target_ds.the_same_dataset_as(source_ds_2_2)


def test_load_by_X_functions(two_empty_temp_db_connections, some_interdeps) -> None:
    """
    Test some different loading functions
    """
    source_conn, target_conn = two_empty_temp_db_connections

    source_path = path_to_dbfile(source_conn)
    target_path = path_to_dbfile(target_conn)

    source_exp1 = Experiment(conn=source_conn)
    source_ds_1_1 = DataSet(conn=source_conn, exp_id=source_exp1.exp_id)

    source_exp2 = Experiment(conn=source_conn)
    source_ds_2_1 = DataSet(conn=source_conn, exp_id=source_exp2.exp_id)

    source_ds_2_2 = DataSet(
        conn=source_conn, exp_id=source_exp2.exp_id, name="customname"
    )

    for ds in (source_ds_1_1, source_ds_2_1, source_ds_2_2):
        ds.set_interdependencies(some_interdeps[1])
        ds.mark_started()
        ds.add_results([{name: 0.0 for name in some_interdeps[1].names}])
        ds.mark_completed()

    extract_runs_into_db(source_path, target_path, source_ds_2_2.run_id)
    extract_runs_into_db(source_path, target_path, source_ds_2_1.run_id)
    extract_runs_into_db(source_path, target_path, source_ds_1_1.run_id)

    test_ds = load_by_guid(source_ds_2_2.guid, target_conn)
    assert source_ds_2_2.the_same_dataset_as(test_ds)

    test_ds = load_by_id(1, target_conn)
    assert source_ds_2_2.the_same_dataset_as(test_ds)

    test_ds = load_by_run_spec(
        captured_run_id=source_ds_2_2.captured_run_id, conn=target_conn
    )
    assert source_ds_2_2.the_same_dataset_as(test_ds)

    assert source_exp2.exp_id == 2

    # this is now the first run in the db so run_id is 1
    target_run_id = 1
    # and the experiment ids will be interchanged.
    target_exp_id = 1

    test_ds = load_by_counter(target_run_id, target_exp_id, target_conn)
    assert source_ds_2_2.the_same_dataset_as(test_ds)


def test_combine_runs(
    two_empty_temp_db_connections, empty_temp_db_connection, some_interdeps
) -> None:
    """
    Test that datasets that are exported in random order from 2 datasets
    can be reloaded by the original captured_run_id and the experiment
    name.
    """
    qc.config.GUID_components.GUID_type = "random_sample"

    source_conn_1, source_conn_2 = two_empty_temp_db_connections
    target_conn = empty_temp_db_connection

    source_1_exp = Experiment(conn=source_conn_1, name="exp1", sample_name="no_sample")
    source_1_datasets = [
        DataSet(conn=source_conn_1, exp_id=source_1_exp.exp_id) for i in range(10)
    ]

    source_2_exp = Experiment(conn=source_conn_2, name="exp2", sample_name="no_sample")

    source_2_datasets = [
        DataSet(conn=source_conn_2, exp_id=source_2_exp.exp_id) for i in range(10)
    ]

    guids_1 = {dataset.guid for dataset in source_1_datasets}
    guids_2 = {dataset.guid for dataset in source_2_datasets}

    guids = guids_1 | guids_2
    assert len(guids) == 20

    source_all_datasets = source_1_datasets + source_2_datasets

    shuffled_datasets = source_all_datasets.copy()
    random.shuffle(shuffled_datasets)

    for ds in source_all_datasets:
        ds.set_interdependencies(some_interdeps[1])
        ds.mark_started()
        ds.add_results([{name: 0.0 for name in some_interdeps[1].names}])
        ds.mark_completed()

    # now let's insert all datasets in random order
    for ds in shuffled_datasets:
        extract_runs_into_db(
            ds.conn.path_to_dbfile, target_conn.path_to_dbfile, ds.run_id
        )

    for ds in source_all_datasets:
        loaded_ds = load_by_run_spec(
            captured_run_id=ds.captured_run_id,
            experiment_name=ds.exp_name,
            conn=target_conn,
        )
        assert ds.the_same_dataset_as(loaded_ds)

    for ds in source_all_datasets:
        loaded_ds = load_by_run_spec(
            captured_run_id=ds.captured_counter,
            experiment_name=ds.exp_name,
            conn=target_conn,
        )
        assert ds.the_same_dataset_as(loaded_ds)

    # Now test that we generate the correct table for the guids above
    # this could be split out into its own test
    # but the test above has the useful side effect of
    # setting up datasets for this test.
    new_guids = [ds.guid for ds in source_all_datasets]

    table = generate_dataset_table(new_guids, conn=target_conn)
    lines = table.split("\n")
    headers = re.split(r"\s+", lines[0].strip())

    cfg = qc.config
    guid_comp = cfg["GUID_components"]

    for i in range(2, len(lines)):
        split_line = re.split(r"\s+", lines[i].strip())
        mydict = {headers[j]: split_line[j] for j in range(len(split_line))}
        ds2 = load_by_guid(new_guids[i - 2], conn=target_conn)
        assert ds2.captured_run_id == int(mydict["captured_run_id"])
        assert ds2.captured_counter == int(mydict["captured_counter"])
        assert ds2.exp_name == mydict["experiment_name"]
        assert ds2.sample_name == mydict["sample_name"]
        assert guid_comp["location"] == int(mydict["location"])
        assert guid_comp["work_station"] == int(mydict["work_station"])


def test_copy_datasets_and_add_new(
    two_empty_temp_db_connections, some_interdeps
) -> None:
    """
    Test that new runs get the correct captured_run_id and captured_counter
    when adding on top of a dataset with partial exports
    """
    source_conn, target_conn = two_empty_temp_db_connections

    source_exp_1 = Experiment(conn=source_conn, name="exp1", sample_name="no_sample")
    source_exp_2 = Experiment(conn=source_conn, name="exp2", sample_name="no_sample")
    source_datasets_1 = [
        DataSet(conn=source_conn, exp_id=source_exp_1.exp_id) for i in range(5)
    ]
    source_datasets_2 = [
        DataSet(conn=source_conn, exp_id=source_exp_2.exp_id) for i in range(5)
    ]
    source_datasets = source_datasets_1 + source_datasets_2

    for ds in source_datasets:
        ds.set_interdependencies(some_interdeps[1])
        ds.mark_started()
        ds.add_results([{name: 0.0 for name in some_interdeps[1].names}])
        ds.mark_completed()

    # now let's insert only some of the datasets
    # and verify that the ids and counters are set correctly
    for ds in source_datasets[-3:]:
        extract_runs_into_db(
            ds.conn.path_to_dbfile, target_conn.path_to_dbfile, ds.run_id
        )

    loaded_datasets = [
        load_by_run_spec(captured_run_id=i, conn=target_conn) for i in range(8, 11)
    ]
    expected_run_ids = [1, 2, 3]
    expected_captured_run_ids = [8, 9, 10]
    expected_counter = [1, 2, 3]
    expected_captured_counter = [3, 4, 5]

    for ds2, eri, ecri, ec, ecc in zip(
        loaded_datasets,
        expected_run_ids,
        expected_captured_run_ids,
        expected_counter,
        expected_captured_counter,
    ):
        assert ds2.run_id == eri
        assert ds2.captured_run_id == ecri
        assert ds2.counter == ec
        assert ds2.captured_counter == ecc

    exp = load_experiment_by_name("exp2", conn=target_conn)

    # add additional runs and verify that the ids and counters increase as
    # expected
    new_datasets = [DataSet(conn=target_conn, exp_id=exp.exp_id) for i in range(3)]

    for ds in new_datasets:
        ds.set_interdependencies(some_interdeps[1])
        ds.mark_started()
        ds.add_results([{name: 0.0 for name in some_interdeps[1].names}])
        ds.mark_completed()

    expected_run_ids = [4, 5, 6]
    expected_captured_run_ids = expected_run_ids
    expected_counter = [4, 5, 6]
    expected_captured_counter = expected_counter

    for ds, eri, ecri, ec, ecc in zip(
        new_datasets,
        expected_run_ids,
        expected_captured_run_ids,
        expected_counter,
        expected_captured_counter,
    ):
        assert ds.run_id == eri
        assert ds.captured_run_id == ecri
        assert ds.counter == ec
        assert ds.captured_counter == ecc


def test_old_versions_not_touched(
    two_empty_temp_db_connections, some_interdeps
) -> None:
    source_conn, target_conn = two_empty_temp_db_connections

    target_path = path_to_dbfile(target_conn)
    source_path = path_to_dbfile(source_conn)

    _, new_v = get_db_version_and_newest_available_version(source_path)

    fixturepath = os.sep.join(tests.dataset.__file__.split(os.sep)[:-1])
    fixturepath = os.path.join(
        fixturepath, "fixtures", "db_files", "version2", "some_runs.db"
    )
    skip_if_no_fixtures(fixturepath)

    # First test that we cannot use an old version as source

    with raise_if_file_changed(fixturepath):
        with pytest.warns(UserWarning) as warning:
            extract_runs_into_db(fixturepath, target_path, 1)
            expected_mssg = (
                "Source DB version is 2, but this "
                f"function needs it to be in version {new_v}. "
                "Run this function again with "
                "upgrade_source_db=True to auto-upgrade "
                "the source DB file."
            )
            assert isinstance(warning[0].message, Warning)
            assert warning[0].message.args[0] == expected_mssg

    # Then test that we cannot use an old version as target

    # first create a run in the new version source
    source_exp = Experiment(conn=source_conn)
    source_ds = DataSet(conn=source_conn, exp_id=source_exp.exp_id)

    source_ds.set_interdependencies(some_interdeps[1])

    source_ds.mark_started()
    source_ds.add_results([{name: 0.0 for name in some_interdeps[1].names}])
    source_ds.mark_completed()

    with raise_if_file_changed(fixturepath):
        with pytest.warns(UserWarning) as warning:
            extract_runs_into_db(source_path, fixturepath, 1)
            expected_mssg = (
                "Target DB version is 2, but this "
                f"function needs it to be in version {new_v}. "
                "Run this function again with "
                "upgrade_target_db=True to auto-upgrade "
                "the target DB file."
            )
            assert isinstance(warning[0].message, Warning)
            assert warning[0].message.args[0] == expected_mssg


def test_experiments_with_NULL_sample_name(
    two_empty_temp_db_connections, some_interdeps
) -> None:
    """
    In older API versions (corresponding to DB version 3),
    users could get away with setting the sample name to None

    This test checks that such an experiment gets correctly recognised and
    is thus not ever re-inserted into the target DB
    """
    source_conn, target_conn = two_empty_temp_db_connections
    source_exp_1 = Experiment(conn=source_conn, name="null_sample_name")

    source_path = path_to_dbfile(source_conn)
    target_path = path_to_dbfile(target_conn)

    # make 5 runs in experiment

    exp_1_run_ids = []
    for _ in range(5):
        source_dataset = DataSet(conn=source_conn, exp_id=source_exp_1.exp_id)
        exp_1_run_ids.append(source_dataset.run_id)

        source_dataset.set_interdependencies(some_interdeps[1])
        source_dataset.mark_started()

        for val in range(10):
            source_dataset.add_results(
                [{name: val for name in some_interdeps[1].names}]
            )
        source_dataset.mark_completed()

    sql = """
          UPDATE experiments
          SET sample_name = NULL
          WHERE exp_id = 1
          """
    source_conn.execute(sql)
    source_conn.commit()

    assert source_exp_1.sample_name is None

    extract_runs_into_db(source_path, target_path, 1, 2, 3, 4, 5)

    assert len(get_experiments(target_conn)) == 1

    extract_runs_into_db(source_path, target_path, 1, 2, 3, 4, 5)

    assert len(get_experiments(target_conn)) == 1

    assert len(Experiment(exp_id=1, conn=target_conn)) == 5


def test_integration_station_and_measurement(
    two_empty_temp_db_connections, inst
) -> None:
    """
    An integration test where the runs in the source DB file are produced
    with the Measurement object and there is a Station as well
    """
    source_conn, target_conn = two_empty_temp_db_connections
    source_path = path_to_dbfile(source_conn)
    target_path = path_to_dbfile(target_conn)

    source_exp = Experiment(conn=source_conn)

    # Set up measurement scenario
    station = Station(inst)

    meas = Measurement(exp=source_exp, station=station)
    meas.register_parameter(inst.back)
    meas.register_parameter(inst.plunger)
    meas.register_parameter(inst.cutter, setpoints=(inst.back, inst.plunger))

    with meas.run() as datasaver:
        for back_v in [1, 2, 3]:
            for plung_v in [-3, -2.5, 0]:
                datasaver.add_result(
                    (inst.back, back_v),
                    (inst.plunger, plung_v),
                    (inst.cutter, back_v + plung_v),
                )

    extract_runs_into_db(source_path, target_path, 1)

    target_ds = DataSet(conn=target_conn, run_id=1)
    assert isinstance(datasaver.dataset, DataSet)
    assert datasaver.dataset.the_same_dataset_as(target_ds)


def test_atomicity(two_empty_temp_db_connections, some_interdeps) -> None:
    """
    Test the atomicity of the transaction by extracting and inserting two
    runs where the second one is not completed. The not completed error must
    roll back any changes to the target
    """
    source_conn, target_conn = two_empty_temp_db_connections

    source_path = path_to_dbfile(source_conn)
    target_path = path_to_dbfile(target_conn)

    # The target file must exist for us to be able to see whether it has
    # changed
    Path(target_path).touch()

    source_exp = Experiment(conn=source_conn)
    source_ds_1 = DataSet(conn=source_conn, exp_id=source_exp.exp_id)
    source_ds_2 = DataSet(conn=source_conn, exp_id=source_exp.exp_id)

    for ds in (source_ds_1, source_ds_2):
        ds.set_interdependencies(some_interdeps[1])
        ds.mark_started()
        ds.add_results([{name: 2.1 for name in some_interdeps[1].names}])

    # importantly, source_ds_2 is NOT marked as completed
    source_ds_1.mark_completed()

    # now check that the target file is untouched
    with raise_if_file_changed(target_path):
        # although the not completed error is a ValueError, we get the
        # RuntimeError from SQLite
        with pytest.raises(RuntimeError):
            extract_runs_into_db(source_path, target_path, 1, 2)


def test_column_mismatch(two_empty_temp_db_connections, some_interdeps, inst) -> None:
    """
    Test insertion of runs with no metadata and no snapshot into a DB already
    containing a run that has both
    """

    source_conn, target_conn = two_empty_temp_db_connections
    source_path = path_to_dbfile(source_conn)
    target_path = path_to_dbfile(target_conn)

    target_exp = Experiment(conn=target_conn)

    # Set up measurement scenario
    station = Station(inst)

    meas = Measurement(exp=target_exp, station=station)
    meas.register_parameter(inst.back)
    meas.register_parameter(inst.plunger)
    meas.register_parameter(inst.cutter, setpoints=(inst.back, inst.plunger))

    with meas.run() as datasaver:
        for back_v in [1, 2, 3]:
            for plung_v in [-3, -2.5, 0]:
                datasaver.add_result(
                    (inst.back, back_v),
                    (inst.plunger, plung_v),
                    (inst.cutter, back_v + plung_v),
                )
    datasaver.dataset.add_metadata("meta_tag", "meta_value")

    Experiment(conn=source_conn)
    source_ds = DataSet(conn=source_conn)
    source_ds.set_interdependencies(some_interdeps[1])

    source_ds.mark_started()
    source_ds.add_results([{name: 2.1 for name in some_interdeps[1].names}])
    source_ds.mark_completed()

    extract_runs_into_db(source_path, target_path, 1)

    # compare
    target_copied_ds = DataSet(conn=target_conn, run_id=2)

    assert target_copied_ds.the_same_dataset_as(source_ds)
