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

    source_exp = Experiment(conn=source_conn)
    source_dataset = DataSet(conn=source_conn)

    with pytest.raises(ValueError, match='Dataset not completed'):
        copy_runs_into_db(source_path, target_path, source_dataset.run_id)

    for ps in some_paramspecs[1].values():
        source_dataset.add_parameter(ps)

    value = 0  # an arbitrary data value
    result = {ps.name: type_casters[ps.type](value)
              for ps in some_paramspecs[1].values()}

    source_dataset.add_result(result)
    source_dataset.mark_complete()

    copy_runs_into_db(source_path, target_path, source_dataset.run_id)

    target_exp = Experiment(conn=target_conn, exp_id=1)

    length1 = len(target_exp)

    # trying to insert the same run again should be a NOOP
    copy_runs_into_db(source_path, target_path, source_dataset.run_id)

    assert len(target_exp) == length1

    target_dataset = DataSet(conn=source_conn, run_id=1)

    # Now make the interesting comparisons: are the target objects the same as
    # the source objects?

    exp_attrs = ['name', 'sample_name', 'started_at', 'finished_at',
                 'format_string']

    ds_attrs = ['name', 'table_name', 'guid', 'number_of_results',
                'counter', 'parameters', 'paramspecs', 'exp_name',
                'sample_name', 'completed', 'snapshot', 'run_timestamp_raw']

    for ds_attr in ds_attrs:
        assert getattr(source_dataset, ds_attr) == getattr(target_dataset, ds_attr)

    # for exp_attr in exp_attrs:
    #     assert getattr(source_exp, exp_attr) == getattr(target_exp, exp_attr)
