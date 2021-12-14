from pathlib import Path
from typing import List

import pytest

import qcodes as qc
from qcodes import load_by_guid, load_or_create_experiment
from qcodes.dataset.data_set_in_memory import DataSetInMem
from qcodes.dataset.descriptions.dependencies import InterDependencies_
from qcodes.dataset.descriptions.param_spec import ParamSpecBase
from qcodes.dataset.sqlite.database import connect


def test_create_dataset_in_memory_explicit_db(empty_temp_db):
    default_db_location = qc.config["core"]["db_location"]

    extra_db_location = str(Path(default_db_location).parent / "extra.db")

    load_or_create_experiment(
        conn=connect(extra_db_location), experiment_name="myexp", sample_name="mysample"
    )

    ds = DataSetInMem._create_new_run(name="foo", path_to_db=str(extra_db_location))

    assert ds.path_to_db == extra_db_location
    assert default_db_location != extra_db_location


def test_empty_ds_parameters(experiment):

    ds = DataSetInMem._create_new_run(name="foo")

    assert ds._parameters is None

    ds._perform_start_actions()
    assert ds._parameters is None

    ds.mark_completed()

    assert ds._parameters is None


def test_write_metadata_to_explicit_db(empty_temp_db):
    default_db_location = qc.config["core"]["db_location"]
    extra_db_location = str(Path(default_db_location).parent / "extra.db")
    load_or_create_experiment(experiment_name="myexp", sample_name="mysample")
    load_or_create_experiment(
        conn=connect(extra_db_location), experiment_name="myexp", sample_name="mysample"
    )
    ds = DataSetInMem._create_new_run(name="foo")
    assert ds._parameters is None
    assert ds.path_to_db == default_db_location
    ds.export("netcdf")
    ds.write_metadata_to_db(path_to_db=extra_db_location)
    loaded_ds = load_by_guid(ds.guid, conn=connect(extra_db_location))

    ds.the_same_dataset_as(loaded_ds)


def test_no_interdeps_raises_in_prepare(experiment):
    ds = DataSetInMem._create_new_run(name="foo")
    with pytest.raises(RuntimeError, match="No parameters supplied"):
        ds.prepare(interdeps=InterDependencies_(), snapshot={})


def test_prepare_twice_raises(experiment):
    ds = DataSetInMem._create_new_run(name="foo")

    pss: List[ParamSpecBase] = []
    for n in range(3):
        pss.append(ParamSpecBase(f"ps{n}", paramtype="numeric"))

    idps = InterDependencies_(dependencies={pss[0]: (pss[1], pss[2])})

    ds.prepare(interdeps=idps, snapshot={})
    with pytest.raises(
        RuntimeError, match="Cannot prepare a dataset that is not pristine."
    ):
        ds.prepare(interdeps=idps, snapshot={})


def test_timestamps(experiment):
    ds = DataSetInMem._create_new_run(name="foo")

    assert ds.run_timestamp() is None
    assert ds.run_timestamp_raw is None

    assert ds.completed_timestamp() is None
    assert ds.completed_timestamp_raw is None

    pss: List[ParamSpecBase] = []
    for n in range(3):
        pss.append(ParamSpecBase(f"ps{n}", paramtype="numeric"))

    idps = InterDependencies_(dependencies={pss[0]: (pss[1], pss[2])})

    ds.prepare(interdeps=idps, snapshot={})

    assert ds.run_timestamp() is not None
    assert ds.run_timestamp_raw is not None

    assert ds.completed_timestamp() is None
    assert ds.completed_timestamp_raw is None

    ds.mark_completed()

    assert ds.run_timestamp() is not None
    assert ds.run_timestamp_raw is not None

    assert ds.completed_timestamp() is not None
    assert ds.completed_timestamp_raw is not None

    ds.mark_completed()


def test_mark_pristine_completed_raises(experiment):
    ds = DataSetInMem._create_new_run(name="foo")

    with pytest.raises(
        RuntimeError, match="Can not mark a dataset as complete before it"
    ):
        ds.mark_completed()


def test_load_from_non_existing_guid(experiment):
    guid = "This is not a guid"
    with pytest.raises(
        RuntimeError, match="Could not find the requested run with GUID"
    ):
        _ = DataSetInMem._load_from_db(conn=experiment.conn, guid=guid)
