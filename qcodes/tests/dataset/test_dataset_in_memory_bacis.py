from pathlib import Path

import qcodes as qc
from qcodes import load_or_create_experiment
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
