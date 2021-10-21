import time
from math import floor

import pytest

from qcodes.dataset.data_set import (
    DataSet,
    load_by_counter,
    load_by_guid,
    load_by_id,
    load_by_run_spec,
    new_data_set,
)
from qcodes.dataset.data_set_info import get_run_attributes
from qcodes.dataset.descriptions.rundescriber import RunDescriber
from qcodes.dataset.experiment_container import new_experiment
from qcodes.dataset.sqlite.queries import (
    get_experiment_attributes_by_exp_id,
    get_guids_from_run_spec,
    get_raw_run_attributes,
    raw_time_to_str_time,
)


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


@pytest.mark.usefixtures("experiment")
def test_get_run_attributes() -> None:
    name = "test-dataset"
    ds = new_data_set(name)
    ds.mark_started()
    ds.mark_completed()
    ds.add_metadata("foo", "bar")

    loaded_raw_attrs = get_raw_run_attributes(ds.conn, ds.guid)
    assert loaded_raw_attrs is not None

    assert loaded_raw_attrs["run_id"] == ds.run_id
    assert loaded_raw_attrs["counter"] == ds.counter
    assert loaded_raw_attrs["captured_counter"] == ds.captured_counter
    assert loaded_raw_attrs["captured_run_id"] == ds.captured_run_id
    assert loaded_raw_attrs["captured_run_id"] == ds.captured_run_id
    assert loaded_raw_attrs["experiment"] == get_experiment_attributes_by_exp_id(
        ds.conn, ds.exp_id
    )
    assert loaded_raw_attrs["experiment"]["exp_id"] == ds.exp_id
    assert loaded_raw_attrs["experiment"]["name"] == ds.exp_name
    assert loaded_raw_attrs["experiment"]["sample_name"] == ds.sample_name
    assert loaded_raw_attrs["name"] == name
    assert loaded_raw_attrs["run_timestamp"] == ds.run_timestamp_raw
    assert loaded_raw_attrs["completed_timestamp"] == ds.completed_timestamp_raw
    assert loaded_raw_attrs["parent_dataset_links"] == "[]"
    assert "interdependencies" in loaded_raw_attrs["run_description"]
    assert loaded_raw_attrs["snapshot"] is None
    assert loaded_raw_attrs["metadata"] == {"foo": "bar"}

    loaded_attrs = get_run_attributes(ds.conn, ds.guid)
    assert loaded_attrs is not None

    assert loaded_attrs["run_id"] == ds.run_id
    assert loaded_attrs["counter"] == ds.counter
    assert loaded_attrs["captured_counter"] == ds.captured_counter
    assert loaded_attrs["captured_run_id"] == ds.captured_run_id
    assert loaded_attrs["captured_run_id"] == ds.captured_run_id
    assert loaded_attrs["experiment"] == get_experiment_attributes_by_exp_id(
        ds.conn, ds.exp_id
    )
    assert loaded_attrs["experiment"]["exp_id"] == ds.exp_id
    assert loaded_attrs["name"] == name
    assert loaded_attrs["run_timestamp"] == raw_time_to_str_time(ds.run_timestamp_raw)
    assert loaded_attrs["completed_timestamp"] == raw_time_to_str_time(
        ds.completed_timestamp_raw
    )
    assert loaded_attrs["parent_dataset_links"] == []
    assert isinstance(loaded_attrs["run_description"], RunDescriber)
    assert loaded_attrs["snapshot"] is None
    assert loaded_attrs["metadata"] == {"foo": "bar"}


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


@pytest.mark.usefixtures('experiment')
def test_load_by_guid(some_interdeps):
    ds = DataSet()
    ds.set_interdependencies(some_interdeps[1])
    ds.mark_started()
    ds.add_results([{'ps1': 1, 'ps2': 2}])

    loaded_ds = load_by_guid(ds.guid)

    assert loaded_ds.the_same_dataset_as(ds)


def test_load_by_run_spec(empty_temp_db, some_interdeps):

    def create_ds_with_exp_id(exp_id):
        ds = DataSet(exp_id=exp_id)
        ds.set_interdependencies(some_interdeps[1])
        ds.mark_started()
        ds.add_results([{'ps1': 1, 'ps2': 2}])
        return ds
    # create 3 experiments that mix two experiment names and two sample names
    exp_names = ["te1", "te2", "te1"]
    sample_names = ["ts1", "ts2", "ts2"]

    exps = [new_experiment(exp_name, sample_name=sample_name)
            for exp_name, sample_name in zip(exp_names, sample_names)]

    created_ds = [create_ds_with_exp_id(exp.exp_id) for exp in exps]

    conn = created_ds[0].conn

    guids = get_guids_from_run_spec(conn=conn)
    assert len(guids) == 3

    # since we are not copying runs from multiple dbs we can always load by
    # captured_run_id and this is equivalent to load_by_id
    for i in range(1, 4):
        loaded_ds = load_by_run_spec(captured_run_id=i,
                                     conn=conn)
        assert loaded_ds.guid == guids[i-1]
        assert loaded_ds.the_same_dataset_as(created_ds[i-1])

    # All the datasets datasets have the same captured counter
    # so we cannot load by that alone
    guids_cc1 = get_guids_from_run_spec(captured_counter=1, conn=conn)
    assert len(guids_cc1) == 3
    with pytest.raises(NameError, match="More than one matching"):
        load_by_run_spec(captured_counter=1)

    # there are two different experiments with exp name "test-experiment1"
    # and thus 2 different datasets with counter=1 and that exp name
    guids_cc1_te1 = get_guids_from_run_spec(captured_counter=1,
                                            experiment_name='te1',
                                            conn=conn)
    assert len(guids_cc1_te1) == 2
    with pytest.raises(NameError, match="More than one matching"):
        load_by_run_spec(captured_counter=1, experiment_name="te1", conn=conn)

    # but for "test-experiment2" there is only one
    guids_cc1_te2 = get_guids_from_run_spec(captured_counter=1,
                                            experiment_name='te2',
                                            conn=conn)
    assert len(guids_cc1_te2) == 1
    loaded_ds = load_by_run_spec(captured_counter=1,
                                 experiment_name="te2",
                                 conn=conn)
    assert loaded_ds.guid == guids_cc1_te2[0]
    assert loaded_ds.the_same_dataset_as(created_ds[1])

    # there are two different experiments with sample name "test_sample2" but
    # different exp names so the counter is not unique
    guids_cc1_ts2 = get_guids_from_run_spec(captured_counter=1,
                                            sample_name='ts2',
                                            conn=conn)
    assert len(guids_cc1_ts2) == 2
    with pytest.raises(NameError, match="More than one matching"):
        load_by_run_spec(captured_counter=1,
                         sample_name="ts2",
                         conn=conn)

    # but for  "test_sample1" there is only one
    guids_cc1_ts1 = get_guids_from_run_spec(captured_counter=1,
                                            sample_name='ts1',
                                            conn=conn)
    assert len(guids_cc1_ts1) == 1
    loaded_ds = load_by_run_spec(captured_counter=1,
                                 sample_name="ts1",
                                 conn=conn)
    assert loaded_ds.the_same_dataset_as(created_ds[0])
    assert loaded_ds.guid == guids_cc1_ts1[0]

    # we can load all 3 if we are specific.
    for i in range(3):
        loaded_ds = load_by_run_spec(captured_counter=1,
                                     experiment_name=exp_names[i],
                                     sample_name=sample_names[i],
                                     conn=conn)
        assert loaded_ds.the_same_dataset_as(created_ds[i])
        assert loaded_ds.guid == guids[i]

    # load a non-existing run
    with pytest.raises(NameError, match="No run matching"):
        load_by_run_spec(captured_counter=10000, sample_name="ts2", conn=conn)

    empty_guid_list = get_guids_from_run_spec(conn=conn,
                                              experiment_name='nosuchexp')
    assert empty_guid_list == []
