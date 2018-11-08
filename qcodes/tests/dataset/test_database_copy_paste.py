from os.path import getmtime
from contextlib import contextmanager

import pytest
import numpy as np

from qcodes.dataset.sqlite_base import connect, get_experiments
from qcodes.dataset.experiment_container import Experiment
from qcodes.dataset.data_set import DataSet, load_by_guid
from qcodes.dataset.database import path_to_dbfile
from qcodes.dataset.database_copy_paste import copy_runs_into_db
from qcodes.tests.dataset.temporary_databases import two_empty_temp_db_connections
from qcodes.tests.dataset.test_descriptions import some_paramspecs
from qcodes.tests.dataset.test_database_creation_and_upgrading import error_caused_by


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
        raise RuntimeError(f'File {path_to_file} was modified.')


def test_basic_copy_paste(two_empty_temp_db_connections, some_paramspecs):
    source_conn, target_conn = two_empty_temp_db_connections

    source_path = path_to_dbfile(source_conn)
    target_path = path_to_dbfile(target_conn)

    type_casters = {'numeric': float,
                    'array': (lambda x: np.array(x) if hasattr(x, '__iter__')
                              else np.array([x])),
                    'text': str}

    source_exp = Experiment(conn=source_conn)
    source_dataset = DataSet(conn=source_conn)

    with pytest.raises(RuntimeError) as excinfo:
        copy_runs_into_db(source_path, target_path, source_dataset.run_id)

    assert error_caused_by(excinfo, ('Dataset not completed. An incomplete '
                                     'dataset can not be copied.'))

    for ps in some_paramspecs[1].values():
        source_dataset.add_parameter(ps)

    for value in range(10):
        result = {ps.name: type_casters[ps.type](value)
                  for ps in some_paramspecs[1].values()}
        source_dataset.add_result(result)

    source_dataset.mark_complete()

    copy_runs_into_db(source_path, target_path, source_dataset.run_id)

    target_exp = Experiment(conn=target_conn, exp_id=1)

    length1 = len(target_exp)

    # trying to insert the same run again should be a NOOP
    with raise_if_file_changed(target_path):
        copy_runs_into_db(source_path, target_path, source_dataset.run_id)

    assert len(target_exp) == length1

    target_dataset = DataSet(conn=source_conn, run_id=1)

    # Now make the interesting comparisons: are the target objects the same as
    # the source objects?

    assert source_dataset.the_same_dataset_as(target_dataset)

    source_data = source_dataset.get_data(*source_dataset.parameters.split(','))
    target_data = target_dataset.get_data(*target_dataset.parameters.split(','))

    assert source_data == target_data

    exp_attrs = ['name', 'sample_name', 'format_string', 'started_at',
                 'finished_at']

    for exp_attr in exp_attrs:
        assert getattr(source_exp, exp_attr) == getattr(target_exp, exp_attr)

    # trying to insert the same run again should be a NOOP
    with raise_if_file_changed(target_path):
        copy_runs_into_db(source_path, target_path, source_dataset.run_id)


def test_correct_experiment_routing(two_empty_temp_db_connections,
                                    some_paramspecs):
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

        for ps in some_paramspecs[2].values():
            source_dataset.add_parameter(ps)

        for val in range(10):
            source_dataset.add_result({ps.name: val
                                       for ps in some_paramspecs[2].values()})
        source_dataset.mark_complete()

    # make a new experiment with 1 run

    source_exp_2 = Experiment(conn=source_conn)
    ds = DataSet(conn=source_conn, exp_id=source_exp_2.exp_id)
    exp_2_run_ids = [ds.run_id]

    for ps in some_paramspecs[2].values():
        ds.add_parameter(ps)

    for val in range(10):
        ds.add_result({ps.name: val for ps in some_paramspecs[2].values()})

    ds.mark_complete()

    source_path = path_to_dbfile(source_conn)
    target_path = path_to_dbfile(target_conn)

    # now copy 2 runs
    copy_runs_into_db(source_path, target_path, *exp_1_run_ids[:2])

    target_exp1 = Experiment(conn=target_conn, exp_id=1)

    assert len(target_exp1) == 2

    # copy two other runs, one of them already in
    copy_runs_into_db(source_path, target_path, *exp_1_run_ids[1:3])

    assert len(target_exp1) == 3

    # insert run from different experiment
    copy_runs_into_db(source_path, target_path, ds.run_id)

    assert len(target_exp1) == 3

    target_exp2 = Experiment(conn=target_conn, exp_id=2)

    assert len(target_exp2) == 1

    # finally insert every single run from experiment 1

    copy_runs_into_db(source_path, target_path, *exp_1_run_ids)

    # check for idempotency once more by inserting all the runs but in another
    # order
    with raise_if_file_changed(target_path):
        copy_runs_into_db(source_path, target_path, *exp_1_run_ids[::-1])

    target_exps = get_experiments(target_conn)

    assert len(target_exps) == 2
    assert len(target_exp1) == 5
    assert len(target_exp2) == 1

    # check that all the datasets match up
    for run_id in exp_1_run_ids + exp_2_run_ids:
        source_ds = DataSet(conn=source_conn, run_id=run_id)
        target_ds = load_by_guid(guid=source_ds.guid, conn=target_conn)

        assert source_ds.the_same_dataset_as(target_ds)

        source_data = source_ds.get_data(*source_ds.parameters.split(','))
        target_data = target_ds.get_data(*target_ds.parameters.split(','))

        assert source_data == target_data


def test_runs_from_different_experiments_raises(two_empty_temp_db_connections,
                                                some_paramspecs):
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

        for ps in some_paramspecs[2].values():
            source_dataset.add_parameter(ps)

        for val in range(10):
            source_dataset.add_result({ps.name: val
                                       for ps in some_paramspecs[2].values()})
        source_dataset.mark_complete()

    # make 5 runs in second experiment

    exp_2_run_ids = []
    for _ in range(5):

        source_dataset = DataSet(conn=source_conn, exp_id=source_exp_2.exp_id)
        exp_2_run_ids.append(source_dataset.run_id)

        for ps in some_paramspecs[2].values():
            source_dataset.add_parameter(ps)

        for val in range(10):
            source_dataset.add_result({ps.name: val
                                       for ps in some_paramspecs[2].values()})
        source_dataset.mark_complete()

    run_ids = exp_1_run_ids + exp_2_run_ids
    source_exp_ids = np.unique([1, 2])
    matchstring = ('Did not receive runs from a single experiment\\. '
                   f'Got runs from experiments {source_exp_ids}')
    # make the matchstring safe to use as a regexp
    matchstring = matchstring.replace('[', '\\[').replace(']', '\\]')
    with pytest.raises(ValueError, match=matchstring):
        copy_runs_into_db(source_path, target_path, *run_ids)
