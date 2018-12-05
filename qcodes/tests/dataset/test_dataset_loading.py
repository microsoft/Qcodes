import time
from math import floor

import pytest

from qcodes.dataset.data_set import (DataSet,
                                     new_data_set,
                                     load_by_guid,
                                     load_by_id,
                                     load_by_counter,
                                     ParamSpec)
from qcodes.dataset.data_export import get_data_by_id
from qcodes.dataset.experiment_container import new_experiment
# pylint: disable=unused-import
from qcodes.tests.dataset.temporary_databases import (empty_temp_db,
                                                      experiment, dataset)
# pylint: disable=unused-import
from qcodes.tests.dataset.test_descriptions import some_paramspecs


@pytest.mark.usefixtures("experiment")
def test_load_by_id():
    ds = new_data_set("test-dataset")
    run_id = ds.run_id
    ds.mark_complete()

    loaded_ds = load_by_id(run_id)
    assert loaded_ds.completed is True
    assert loaded_ds.exp_id == 1

    ds = new_data_set("test-dataset-unfinished")
    run_id = ds.run_id

    loaded_ds = load_by_id(run_id)
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

    assert loaded_ds.completed is False

    ds.mark_complete()
    loaded_ds = load_by_counter(exp.exp_id, 1)

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
    t_after_data_set = time.time()

    actual_run_timestamp_raw = ds.run_timestamp_raw

    assert t_before_data_set <= actual_run_timestamp_raw <= t_after_data_set


@pytest.mark.usefixtures("empty_temp_db")
def test_run_timestamp_with_default_format():
    _ = new_experiment(name="for_loading", sample_name="no_sample")

    t_before_data_set = time.time()
    ds = new_data_set("my_first_ds")
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
    ds.mark_complete()
    t_after_complete = time.time()

    actual_completed_timestamp_raw = ds.completed_timestamp_raw

    assert t_before_complete \
           <= actual_completed_timestamp_raw \
           <= t_after_complete


@pytest.mark.usefixtures("empty_temp_db")
def test_completed_timestamp_for_not_completed_dataset():
    _ = new_experiment(name="for_loading", sample_name="no_sample")
    ds = new_data_set("my_first_ds")

    assert False is ds.completed

    assert None is ds.completed_timestamp_raw

    assert None is ds.completed_timestamp()


@pytest.mark.usefixtures("empty_temp_db")
def test_completed_timestamp_with_default_format():
    _ = new_experiment(name="for_loading", sample_name="no_sample")
    ds = new_data_set("my_first_ds")

    t_before_complete = time.time()
    ds.mark_complete()
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
    Test if the values of the setpoints/dependent parameters is dependent on
    the order of the `depends_on` value. This sounds far fetch but was
    actually the case before #1250.
    """
    indepA = ParamSpec('indep1', "numeric")
    indepB = ParamSpec('indep2', "numeric")
    depAB = ParamSpec('depAB', "numeric", depends_on=[indepA, indepB])
    depBA = ParamSpec('depBA', "numeric", depends_on=[indepB, indepA])
    dataset.add_parameter(indepA)
    dataset.add_parameter(indepB)
    dataset.add_parameter(depAB)
    dataset.add_parameter(depBA)

    dataset.add_result({'depAB': 12,
                        'indep2': 2,
                        'indep1': 1})

    dataset.add_result({'depBA': 21,
                        'indep2': 2,
                        'indep1': 1})
    dataset.mark_complete()

    data = get_data_by_id(dataset.run_id)
    data_dict = {el['name']: el['data'] for el in data[0]}
    assert data_dict['indep1'] == 1
    assert data_dict['indep2'] == 2

    data_dict = {el['name']: el['data'] for el in data[1]}
    assert data_dict['indep1'] == 1
    assert data_dict['indep2'] == 2


@pytest.mark.usefixtures('experiment')
def test_load_by_guid(some_paramspecs):
    paramspecs = some_paramspecs[2]
    ds = DataSet()
    ds.add_parameter(paramspecs['ps1'])
    ds.add_parameter(paramspecs['ps2'])
    ds.add_result({'ps1': 1, 'ps2': 2})

    loaded_ds = load_by_guid(ds.guid)

    assert loaded_ds.the_same_dataset_as(ds)
