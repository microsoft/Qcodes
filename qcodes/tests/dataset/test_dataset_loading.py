
import time
from math import floor

import pytest

from qcodes.dataset.data_set import (DataSet,
                                     new_data_set,
                                     load_by_guid,
                                     load_by_id,
                                     load_by_counter,
                                     load_by_run_spec)
from qcodes.dataset.descriptions.param_spec import ParamSpecBase
from qcodes.dataset.descriptions.dependencies import InterDependencies_
from qcodes.dataset.data_export import get_data_by_id
from qcodes.dataset.experiment_container import new_experiment
# pylint: disable=unused-import
from qcodes.tests.dataset.temporary_databases import (empty_temp_db,
                                                      experiment, dataset)
# pylint: disable=unused-import
from qcodes.tests.dataset.test_dependencies import some_interdeps


@pytest.mark.usefixtures("experiment")
def test_load_by_id():
    ds = new_data_set("test-dataset")
    run_id = ds.run_id
    ds.mark_started()
    ds.mark_completed()

    loaded_ds = load_by_id(run_id)
    assert ds.started is True
    assert ds.pristine is False
    assert ds.running is False
    assert loaded_ds.completed is True
    assert loaded_ds.exp_id == 1

    ds = new_data_set("test-dataset-unfinished")
    run_id = ds.run_id

    loaded_ds = load_by_id(run_id)
    assert ds.pristine is True
    assert ds.running is False
    assert ds.started is False
    assert loaded_ds.completed is False
    assert loaded_ds.exp_id == 1

    # let's take a run number that is not in the temporary test database file
    non_existing_run_id = run_id + 1
    with pytest.raises(ValueError, match=f"Run with run_id "
                                         f"{non_existing_run_id} does not "
                                         f"exist in the database"):
        _ = load_by_id(non_existing_run_id)


@pytest.mark.usefixtures("empty_temp_db")
def test_load_by_counter():
    exp = new_experiment(name="for_loading", sample_name="no_sample")
    ds = new_data_set("my_first_ds")

    loaded_ds = load_by_counter(exp.exp_id, 1)

    assert loaded_ds.pristine is True
    assert loaded_ds.running is False
    assert loaded_ds.started is False
    assert loaded_ds.completed is False

    ds.mark_started()
    ds.mark_completed()

    loaded_ds = load_by_counter(exp.exp_id, 1)

    assert loaded_ds.pristine is False
    assert loaded_ds.started is True
    assert loaded_ds.running is False
    assert loaded_ds.completed is True


@pytest.mark.usefixtures("empty_temp_db")
def test_experiment_info_in_dataset():
    exp = new_experiment(name="for_loading", sample_name="no_sample")
    ds = new_data_set("my_first_ds")

    assert ds.exp_id == exp.exp_id
    assert ds.exp_name == exp.name
    assert ds.sample_name == exp.sample_name


@pytest.mark.usefixtures("empty_temp_db")
def test_run_timestamp():
    _ = new_experiment(name="for_loading", sample_name="no_sample")

    t_before_data_set = time.time()
    ds = new_data_set("my_first_ds")
    ds.mark_started()
    t_after_data_set = time.time()

    actual_run_timestamp_raw = ds.run_timestamp_raw

    assert t_before_data_set <= actual_run_timestamp_raw <= t_after_data_set


@pytest.mark.usefixtures("empty_temp_db")
def test_run_timestamp_with_default_format():
    _ = new_experiment(name="for_loading", sample_name="no_sample")

    t_before_data_set = time.time()
    ds = new_data_set("my_first_ds")
    ds.mark_started()
    t_after_data_set = time.time()

    # Note that here we also test the default format of `run_timestamp`
    actual_run_timestamp_raw = time.mktime(
        time.strptime(ds.run_timestamp(), "%Y-%m-%d %H:%M:%S"))

    # Note that because the default format precision is 1 second, we add this
    # second to the right side of the comparison
    t_before_data_set_secs = floor(t_before_data_set)
    t_after_data_set_secs = floor(t_after_data_set)
    assert t_before_data_set_secs \
           <= actual_run_timestamp_raw \
           <= t_after_data_set_secs + 1


@pytest.mark.usefixtures("empty_temp_db")
def test_completed_timestamp():
    _ = new_experiment(name="for_loading", sample_name="no_sample")
    ds = new_data_set("my_first_ds")

    t_before_complete = time.time()
    ds.mark_started()
    ds.mark_completed()
    t_after_complete = time.time()

    actual_completed_timestamp_raw = ds.completed_timestamp_raw

    assert t_before_complete \
           <= actual_completed_timestamp_raw \
           <= t_after_complete


@pytest.mark.usefixtures("empty_temp_db")
def test_completed_timestamp_for_not_completed_dataset():
    _ = new_experiment(name="for_loading", sample_name="no_sample")
    ds = new_data_set("my_first_ds")

    assert ds.pristine is True
    assert ds.started is False
    assert ds.running is False
    assert ds.completed is False

    assert ds.completed_timestamp_raw is None

    assert ds.completed_timestamp() is None


@pytest.mark.usefixtures("empty_temp_db")
def test_completed_timestamp_with_default_format():
    _ = new_experiment(name="for_loading", sample_name="no_sample")
    ds = new_data_set("my_first_ds")

    t_before_complete = time.time()
    ds.mark_started()
    ds.mark_completed()
    t_after_complete = time.time()

    # Note that here we also test the default format of `completed_timestamp`
    actual_completed_timestamp_raw = time.mktime(
        time.strptime(ds.completed_timestamp(), "%Y-%m-%d %H:%M:%S"))

    # Note that because the default format precision is 1 second, we add this
    # second to the right side of the comparison
    t_before_complete_secs = floor(t_before_complete)
    t_after_complete_secs = floor(t_after_complete)
    assert t_before_complete_secs \
           <= actual_completed_timestamp_raw \
           <= t_after_complete_secs + 1


def test_get_data_by_id_order(dataset):
    """
    Test that the added values of setpoints end up associated with the correct
    setpoint parameter, irrespective of the ordering of those setpoint
    parameters
    """
    indepA = ParamSpecBase('indep1', "numeric")
    indepB = ParamSpecBase('indep2', "numeric")
    depAB = ParamSpecBase('depAB', "numeric")
    depBA = ParamSpecBase('depBA', "numeric")

    idps = InterDependencies_(
        dependencies={depAB: (indepA, indepB), depBA: (indepB, indepA)})

    dataset.set_interdependencies(idps)

    dataset.mark_started()

    dataset.add_result({'depAB': 12,
                        'indep2': 2,
                        'indep1': 1})

    dataset.add_result({'depBA': 21,
                        'indep2': 2,
                        'indep1': 1})
    dataset.mark_completed()

    data = get_data_by_id(dataset.run_id)
    data_dict = {el['name']: el['data'] for el in data[0]}
    assert data_dict['indep1'] == 1
    assert data_dict['indep2'] == 2

    data_dict = {el['name']: el['data'] for el in data[1]}
    assert data_dict['indep1'] == 1
    assert data_dict['indep2'] == 2


@pytest.mark.usefixtures('experiment')
def test_load_by_guid(some_interdeps):
    ds = DataSet()
    ds.set_interdependencies(some_interdeps[1])
    ds.mark_started()
    ds.add_result({'ps1': 1, 'ps2': 2})

    loaded_ds = load_by_guid(ds.guid)

    assert loaded_ds.the_same_dataset_as(ds)


def test_load_by_run_spec(empty_temp_db, some_interdeps):

    def create_ds_with_exp_id(exp_id):
        ds = DataSet(exp_id=exp_id)
        ds.set_interdependencies(some_interdeps[1])
        ds.mark_started()
        ds.add_result({'ps1': 1, 'ps2': 2})
        return ds
    # create 3 experiments that mixed two experiment names and two sample names
    exp_names = ["test-experiment1", "test-experiment2", "test-experiment1"]
    sample_names = ["test-sample1", "test_sample2", "test_sample2"]

    exps = [new_experiment(exp_name, sample_name=sample_name)
            for exp_name, sample_name in zip(exp_names, sample_names)]

    created_ds = [create_ds_with_exp_id(exp.exp_id) for exp in exps]

    conn = created_ds[0].conn

    # since we are not merging dbs we can always load by captured_run_id
    # this is equivalent to load_by_id
    for i in range(1, 4):
        loaded_ds = load_by_run_spec(captured_run_id=i,
                                     conn=conn)
        assert loaded_ds.the_same_dataset_as(created_ds[i-1])

    # All the datasets will have the same counter (since the experiments are
    # different so this will fail.
    with pytest.raises(NameError):
        load_by_run_spec(captured_counter=1)

    # there are two different experiments with exp name "test-experiment1" but
    # different sample names so the counter is not unique
    with pytest.raises(NameError):
        load_by_run_spec(captured_counter=1, experiment_name="test-experiment1")

    # but for  "test-experiment2" it is
    loaded_ds = load_by_run_spec(captured_counter=1,
                                 experiment_name="test-experiment2")
    assert loaded_ds.the_same_dataset_as(created_ds[1])

    # there are two different experiments with sample name "test_sample2" but
    # different exp names so the counter is not unique
    with pytest.raises(NameError):
        load_by_run_spec(captured_counter=1, sample_name="test_sample2")

    # but for  "test_sample1" it is
    loaded_ds = load_by_run_spec(captured_counter=1,
                                 sample_name="test-sample1")
    assert loaded_ds.the_same_dataset_as(created_ds[0])

    # we can load all 3 if we are specific.
    for i in range(3):
        loaded_ds = load_by_run_spec(captured_counter=1,
                                     experiment_name=exp_names[i],
                                     sample_name=sample_names[i])
        assert loaded_ds.the_same_dataset_as(created_ds[i])
