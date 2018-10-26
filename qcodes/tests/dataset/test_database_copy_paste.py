import pytest

from qcodes.dataset.sqlite_base import connect
from qcodes.dataset.experiment_container import Experiment
from qcodes.dataset.data_set import DataSet
from qcodes.dataset.database_copy_paste import copy_runs_into_db
from qcodes.tests.dataset.temporary_databases import two_empty_temp_dbs


def test_basic_copy_paste(two_empty_temp_dbs):
    source_path, target_path = two_empty_temp_dbs

    source_conn = connect(source_path)
    target_conn = connect(target_path)

    exp = Experiment(conn=source_conn)
    dataset = DataSet(conn=source_conn)

    with pytest.raises(ValueError, match='Dataset not completed'):
        copy_runs_into_db(source_path, target_path, dataset.run_id)


