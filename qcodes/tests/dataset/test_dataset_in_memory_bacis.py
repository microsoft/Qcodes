from pathlib import Path

import qcodes as qc
from qcodes import load_by_guid, load_or_create_experiment, new_data_set
from qcodes.dataset.data_set_in_memory import DataSetInMem
from qcodes.dataset.sqlite.database import connect


def test_create_dataset_in_memory_explicit_db(empty_temp_db):
    default_db_location = qc.config["core"]["db_location"]

    extra_db_location = str(Path(default_db_location).parent / "extra.db")

    load_or_create_experiment(
        conn=connect(extra_db_location), experiment_name="myexp", sample_name="mysample"
    )

    ds = DataSetInMem.create_new_run(name="foo", path_to_db=str(extra_db_location))

    assert ds.path_to_db == extra_db_location
    assert default_db_location != extra_db_location


def test_empty_ds_parameters(experiment):

    ds = DataSetInMem.create_new_run(name="foo")

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
    ds = DataSetInMem.create_new_run(name="foo")
    assert ds._parameters is None
    assert ds.path_to_db == default_db_location
    ds.write_metadata_to_db(path_to_db=extra_db_location)
    loaded_ds = load_by_guid(ds.guid, conn=connect(extra_db_location))

    ds.the_same_dataset_as(loaded_ds)
