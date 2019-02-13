from os.path import getmtime
from contextlib import contextmanager
import re
import os
from pathlib import Path

import pytest
import numpy as np

import qcodes.tests.dataset
from qcodes.dataset.sqlite_base import get_experiments
from qcodes.dataset.experiment_container import Experiment
from qcodes.dataset.data_set import (DataSet, load_by_guid, load_by_counter,
                                     load_by_id)
from qcodes.dataset.database import path_to_dbfile
from qcodes.dataset.database_extract_runs import extract_runs_into_db
from qcodes.tests.dataset.temporary_databases import two_empty_temp_db_connections
from qcodes.tests.dataset.test_descriptions import some_paramspecs
from qcodes.tests.common import error_caused_by
from qcodes.dataset.measurements import Measurement
from qcodes import Station
from qcodes.tests.instrument_mocks import DummyInstrument


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


@pytest.fixture(scope='function')
def inst():
    """
    Dummy instrument for testing, ensuring that it's instance gets closed
    and removed from the global register of instruments, which, if not done,
    make break other tests
    """
    inst = DummyInstrument('inst', gates=['back', 'plunger', 'cutter'])
    yield inst
    inst.close()


def test_missing_runs_raises(two_empty_temp_db_connections, some_paramspecs):
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

        for ps in some_paramspecs[2].values():
            source_dataset.add_parameter(ps)

        for val in range(10):
            source_dataset.add_result({ps.name: val
                                       for ps in some_paramspecs[2].values()})
        source_dataset.mark_complete()

    source_path = path_to_dbfile(source_conn)
    target_path = path_to_dbfile(target_conn)

    run_ids = [1, 8, 5, 3, 2, 4, 4, 4, 7, 8]
    wrong_ids = [8, 7, 8]

    expected_err = ("Error: not all run_ids exist in the source database. "
                    "The following run(s) is/are not present: "
                    f"{wrong_ids}")

    with pytest.raises(ValueError, match=re.escape(expected_err)):
        extract_runs_into_db(source_path, target_path, *run_ids)


def test_basic_extraction(two_empty_temp_db_connections, some_paramspecs):
    source_conn, target_conn = two_empty_temp_db_connections

    source_path = path_to_dbfile(source_conn)
    target_path = path_to_dbfile(target_conn)

    type_casters = {'numeric': float,
                    'array': (lambda x: np.array(x) if hasattr(x, '__iter__')
                              else np.array([x])),
                    'text': str}

    source_exp = Experiment(conn=source_conn)
    source_dataset = DataSet(conn=source_conn, name="basic_copy_paste_name")

    with pytest.raises(RuntimeError) as excinfo:
        extract_runs_into_db(source_path, target_path, source_dataset.run_id)

    assert error_caused_by(excinfo, ('Dataset not completed. An incomplete '
                                     'dataset can not be copied. The '
                                     'incomplete dataset has GUID: '
                                     f'{source_dataset.guid} and run_id: '
                                     f'{source_dataset.run_id}'))

    for ps in some_paramspecs[1].values():
        source_dataset.add_parameter(ps)

    for value in range(10):
        result = {ps.name: type_casters[ps.type](value)
                  for ps in some_paramspecs[1].values()}
        source_dataset.add_result(result)

    source_dataset.add_metadata('goodness', 'fair')
    source_dataset.add_metadata('test', True)

    source_dataset.mark_complete()

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

    source_data = source_dataset.get_data(*source_dataset.parameters.split(','))
    target_data = target_dataset.get_data(*target_dataset.parameters.split(','))

    assert source_data == target_data

    exp_attrs = ['name', 'sample_name', 'format_string', 'started_at',
                 'finished_at']

    for exp_attr in exp_attrs:
        assert getattr(source_exp, exp_attr) == getattr(target_exp, exp_attr)

    # trying to insert the same run again should be a NOOP
    with raise_if_file_changed(target_path):
        extract_runs_into_db(source_path, target_path, source_dataset.run_id)


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
    ds = DataSet(conn=source_conn, exp_id=source_exp_2.exp_id, name="lala")
    exp_2_run_ids = [ds.run_id]

    for ps in some_paramspecs[2].values():
        ds.add_parameter(ps)

    for val in range(10):
        ds.add_result({ps.name: val for ps in some_paramspecs[2].values()})

    ds.mark_complete()

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
        extract_runs_into_db(source_path, target_path, *run_ids)


def test_extracting_dataless_run(two_empty_temp_db_connections,
                                 some_paramspecs):
    """
    Although contrived, it could happen that a run with no data is extracted
    """
    source_conn, target_conn = two_empty_temp_db_connections

    source_path = path_to_dbfile(source_conn)
    target_path = path_to_dbfile(target_conn)

    Experiment(conn=source_conn)

    source_ds = DataSet(conn=source_conn)

    source_ds.mark_complete()

    extract_runs_into_db(source_path, target_path, source_ds.run_id)

    loaded_ds = DataSet(conn=target_conn, run_id=1)

    assert loaded_ds.the_same_dataset_as(source_ds)


def test_result_table_naming_and_run_id(two_empty_temp_db_connections,
                                        some_paramspecs):
    """
    Check that a correct result table name is given and that a correct run_id
    is assigned
    """
    source_conn, target_conn = two_empty_temp_db_connections

    source_path = path_to_dbfile(source_conn)
    target_path = path_to_dbfile(target_conn)

    source_exp1 = Experiment(conn=source_conn)
    source_ds_1_1 = DataSet(conn=source_conn, exp_id=source_exp1.exp_id)
    for ps in some_paramspecs[2].values():
        source_ds_1_1.add_parameter(ps)
    source_ds_1_1.add_result({ps.name: 0.0
                              for ps in some_paramspecs[2].values()})
    source_ds_1_1.mark_complete()

    source_exp2 = Experiment(conn=source_conn)
    source_ds_2_1 = DataSet(conn=source_conn, exp_id=source_exp2.exp_id)
    for ps in some_paramspecs[2].values():
        source_ds_2_1.add_parameter(ps)
    source_ds_2_1.add_result({ps.name: 0.0
                              for ps in some_paramspecs[2].values()})
    source_ds_2_1.mark_complete()
    source_ds_2_2 = DataSet(conn=source_conn,
                            exp_id=source_exp2.exp_id,
                            name="customname")
    for ps in some_paramspecs[2].values():
        source_ds_2_2.add_parameter(ps)
    source_ds_2_2.add_result({ps.name: 0.0
                              for ps in some_paramspecs[2].values()})
    source_ds_2_2.mark_complete()

    extract_runs_into_db(source_path, target_path, source_ds_2_2.run_id)

    # The target ds ought to have a runs table "customname-1-1"
    # and ought to be the same dataset as its "ancestor"
    target_ds = DataSet(conn=target_conn, run_id=1)

    assert target_ds.table_name == "customname-1-1"
    assert target_ds.the_same_dataset_as(source_ds_2_2)


def test_load_by_X_functions(two_empty_temp_db_connections,
                             some_paramspecs):
    """
    Test some different loading functions
    """
    source_conn, target_conn = two_empty_temp_db_connections

    source_path = path_to_dbfile(source_conn)
    target_path = path_to_dbfile(target_conn)

    source_exp1 = Experiment(conn=source_conn)
    source_ds_1_1 = DataSet(conn=source_conn, exp_id=source_exp1.exp_id)
    for ps in some_paramspecs[2].values():
        source_ds_1_1.add_parameter(ps)
    source_ds_1_1.add_result({ps.name: 0.0
                              for ps in some_paramspecs[2].values()})
    source_ds_1_1.mark_complete()

    source_exp2 = Experiment(conn=source_conn)
    source_ds_2_1 = DataSet(conn=source_conn, exp_id=source_exp2.exp_id)
    for ps in some_paramspecs[2].values():
        source_ds_2_1.add_parameter(ps)
    source_ds_2_1.add_result({ps.name: 0.0
                              for ps in some_paramspecs[2].values()})
    source_ds_2_1.mark_complete()
    source_ds_2_2 = DataSet(conn=source_conn,
                            exp_id=source_exp2.exp_id,
                            name="customname")
    for ps in some_paramspecs[2].values():
        source_ds_2_2.add_parameter(ps)
    source_ds_2_2.add_result({ps.name: 0.0
                              for ps in some_paramspecs[2].values()})
    source_ds_2_2.mark_complete()

    extract_runs_into_db(source_path, target_path, source_ds_2_2.run_id)

    test_ds = load_by_guid(source_ds_2_2.guid, target_conn)
    assert source_ds_2_2.the_same_dataset_as(test_ds)

    test_ds = load_by_id(1, target_conn)
    assert source_ds_2_2.the_same_dataset_as(test_ds)

    test_ds = load_by_counter(1, 1, target_conn)
    assert source_ds_2_2.the_same_dataset_as(test_ds)


def test_old_versions_not_touched(two_empty_temp_db_connections,
                                  some_paramspecs):

    source_conn, target_conn = two_empty_temp_db_connections

    target_path = path_to_dbfile(target_conn)
    source_path = path_to_dbfile(source_conn)

    fixturepath = os.sep.join(qcodes.tests.dataset.__file__.split(os.sep)[:-1])
    fixturepath = os.path.join(fixturepath,
                               'fixtures', 'db_files', 'version2',
                               'some_runs.db')
    if not os.path.exists(fixturepath):
        pytest.skip("No db-file fixtures found. You can generate test db-files"
                    " using the scripts in the legacy_DB_generation folder")

    # First test that we can not use an old version as source

    with raise_if_file_changed(fixturepath):
        with pytest.warns(UserWarning) as warning:
            extract_runs_into_db(fixturepath, target_path, 1)
            expected_mssg = ('Source DB version is 2, but this '
                             'function needs it to be in version 4. '
                             'Run this function again with '
                             'upgrade_source_db=True to auto-upgrade '
                             'the source DB file.')
            assert warning[0].message.args[0] == expected_mssg

    # Then test that we can not use an old version as target

    # first create a run in the new version source
    source_exp = Experiment(conn=source_conn)
    source_ds = DataSet(conn=source_conn, exp_id=source_exp.exp_id)

    for ps in some_paramspecs[2].values():
        source_ds.add_parameter(ps)
    source_ds.add_result({ps.name: 0.0
                              for ps in some_paramspecs[2].values()})
    source_ds.mark_complete()

    with raise_if_file_changed(fixturepath):
        with pytest.warns(UserWarning) as warning:
            extract_runs_into_db(source_path, fixturepath, 1)
            expected_mssg = ('Target DB version is 2, but this '
                             'function needs it to be in version 4. '
                             'Run this function again with '
                             'upgrade_target_db=True to auto-upgrade '
                             'the target DB file.')
            assert warning[0].message.args[0] == expected_mssg


def test_experiments_with_NULL_sample_name(two_empty_temp_db_connections,
                                           some_paramspecs):
    """
    In older API versions (corresponding to DB version 3),
    users could get away with setting the sample name to None

    This test checks that such an experiment gets correctly recognised and
    is thus not ever re-inserted into the target DB
    """
    source_conn, target_conn = two_empty_temp_db_connections
    source_exp_1 = Experiment(conn=source_conn, name='null_sample_name')

    source_path = path_to_dbfile(source_conn)
    target_path = path_to_dbfile(target_conn)

    # make 5 runs in experiment

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


def test_integration_station_and_measurement(two_empty_temp_db_connections,
                                             inst):
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
                datasaver.add_result((inst.back, back_v),
                                     (inst.plunger, plung_v),
                                     (inst.cutter, back_v+plung_v))

    extract_runs_into_db(source_path, target_path, 1)

    target_ds = DataSet(conn=target_conn, run_id=1)

    assert datasaver.dataset.the_same_dataset_as(target_ds)


def test_atomicity(two_empty_temp_db_connections, some_paramspecs):
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
    for ps in some_paramspecs[2].values():
        source_ds_1.add_parameter(ps)
    source_ds_1.add_result({ps.name: 2.1
                            for ps in some_paramspecs[2].values()})
    source_ds_1.mark_complete()

    source_ds_2 = DataSet(conn=source_conn, exp_id=source_exp.exp_id)
    for ps in some_paramspecs[2].values():
        source_ds_2.add_parameter(ps)
    source_ds_2.add_result({ps.name: 2.1
                            for ps in some_paramspecs[2].values()})
    # This dataset is NOT marked as completed

    # now check that the target file is untouched
    with raise_if_file_changed(target_path):
        # although the not completed error is a ValueError, we get the
        # RuntimeError from SQLite
        with pytest.raises(RuntimeError):
            extract_runs_into_db(source_path, target_path, 1, 2)


def test_column_mismatch(two_empty_temp_db_connections, some_paramspecs, inst):
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
                datasaver.add_result((inst.back, back_v),
                                     (inst.plunger, plung_v),
                                     (inst.cutter, back_v+plung_v))
    datasaver.dataset.add_metadata('meta_tag', 'meta_value')

    Experiment(conn=source_conn)
    source_ds = DataSet(conn=source_conn)
    for ps in some_paramspecs[2].values():
        source_ds.add_parameter(ps)
    source_ds.add_result({ps.name: 2.1
                          for ps in some_paramspecs[2].values()})
    source_ds.mark_complete()

    extract_runs_into_db(source_path, target_path, 1)

    # compare
    target_copied_ds = DataSet(conn=target_conn, run_id=2)

    assert target_copied_ds.the_same_dataset_as(source_ds)
