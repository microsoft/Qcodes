import pytest
import numpy as np

from qcodes.dataset.sqlite_base import connect
from qcodes.dataset.experiment_container import Experiment
from qcodes.dataset.data_set import DataSet
from qcodes.dataset.database_copy_paste import copy_runs_into_db
from qcodes.tests.dataset.temporary_databases import two_empty_temp_dbs
from qcodes.tests.dataset.test_descriptions import some_paramspecs


def test_basic_copy_paste(two_empty_temp_dbs, some_paramspecs):
    source_path, target_path = two_empty_temp_dbs

    type_casters = {'numeric': float,
                    'array': (lambda x: np.array(x) if hasattr(x, '__iter__')
                              else np.array([x])),
                    'text': str}

    source_conn = connect(source_path)
    target_conn = connect(target_path)

    exp = Experiment(conn=source_conn)
    dataset = DataSet(conn=source_conn)

    with pytest.raises(ValueError, match='Dataset not completed'):
        copy_runs_into_db(source_path, target_path, dataset.run_id)

    for ps in some_paramspecs[1].values():
        dataset.add_parameter(ps)

    value = 0  # an arbitrary data value
    result = {ps.name: type_casters[ps.type](value)
              for ps in some_paramspecs[1].values()}

    dataset.add_result(result)
    dataset.mark_complete()

    copy_runs_into_db(source_path, target_path, dataset.run_id)
