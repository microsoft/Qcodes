import json

import numpy as np

from qcodes.dataset import load_by_guid, load_from_netcdf
from qcodes.dataset.data_set import DataSet
from qcodes.dataset.data_set_in_memory import DataSetInMem
from qcodes.dataset.experiment_container import Experiment
from qcodes.dataset.sqlite.connection import path_to_dbfile


def test_basic_roundtrip(
    two_empty_temp_db_connections, some_interdeps, tmp_path
) -> None:
    """
    Test that we can export from one db and import into another in
    the most basic form.
    """
    source_conn, target_conn = two_empty_temp_db_connections

    source_exp_1 = Experiment(conn=source_conn)

    source_path = path_to_dbfile(source_conn)
    target_path = path_to_dbfile(target_conn)

    # make 5 runs in first experiment
    source_datasets = []

    for i in range(5):
        source_dataset = DataSetInMem._create_new_run(
            name=f"ds{i}",
            exp_id=source_exp_1.exp_id,
            path_to_db=source_path,
        )
        source_datasets.append(source_dataset)
        source_dataset._set_interdependencies(some_interdeps[1])

        source_dataset._perform_start_actions()

        for val in range(10):
            source_dataset._enqueue_results(
                {name: np.array([val]) for name in some_interdeps[1].names}
            )
        source_dataset.mark_completed()
        source_dataset.export(export_type="netcdf", path=str(tmp_path))

    loaded_datasets = [
        load_from_netcdf(
            json.loads(ds.metadata["export_info"])["export_paths"]["nc"], target_path
        )
        for ds in source_datasets
    ]
    _assert_before_writing_metadata(source_datasets, loaded_datasets)

    for loaded_ds in loaded_datasets:
        loaded_ds.write_metadata_to_db()

    reloaded_ds = [load_by_guid(ds.guid, target_conn) for ds in loaded_datasets]

    _assert_after_writing_metadata(
        source_datasets,
        loaded_datasets,
        reloaded_ds,
        offset_total_ds=0,
        offset_ds_in_exp=0,
        offset_exps=0,
    )


def test_roundtrip_to_db_with_existing_meas(
    two_empty_temp_db_connections, some_interdeps, tmp_path
) -> None:
    """
    Test that we can export from one db and import into another in
    the most basic form.
    """

    n_extra_runs = 4
    n_extra_experiments = 1

    source_conn, target_conn = two_empty_temp_db_connections

    source_exp_1 = Experiment(conn=source_conn)
    target_exp_1 = Experiment(
        conn=target_conn, name="some_other_exp", sample_name="some_other_sample"
    )

    source_path = path_to_dbfile(source_conn)
    target_path = path_to_dbfile(target_conn)

    # make 5 runs in first experiment
    source_datasets = []

    # write datasets to test on

    for i in range(5):
        source_dataset = DataSetInMem._create_new_run(
            name=f"ds{i}",
            exp_id=source_exp_1.exp_id,
            path_to_db=source_path,
        )
        source_datasets.append(source_dataset)
        source_dataset._set_interdependencies(some_interdeps[1])

        source_dataset._perform_start_actions()

        for val in range(10):
            source_dataset._enqueue_results(
                {name: np.array([val]) for name in some_interdeps[1].names}
            )
        source_dataset.mark_completed()
        source_dataset.export(export_type="netcdf", path=str(tmp_path))

    extra_datasets = []

    for i in range(n_extra_runs):
        extra_dataset = DataSet(conn=target_conn, exp_id=target_exp_1.exp_id)

        extra_datasets.append(extra_dataset)
        extra_dataset.set_interdependencies(some_interdeps[1])

        extra_dataset.mark_started()

        for val in range(10):
            extra_dataset._enqueue_results(
                {name: np.array([val]) for name in some_interdeps[1].names}
            )
        extra_dataset.mark_completed()

    loaded_datasets = [
        load_from_netcdf(
            json.loads(ds.metadata["export_info"])["export_paths"]["nc"], target_path
        )
        for ds in source_datasets
    ]

    _assert_before_writing_metadata(source_datasets, loaded_datasets)

    for loaded_ds in loaded_datasets:
        loaded_ds.write_metadata_to_db()

    reloaded_ds = [load_by_guid(ds.guid, target_conn) for ds in loaded_datasets]

    _assert_after_writing_metadata(
        source_datasets,
        loaded_datasets,
        reloaded_ds,
        offset_total_ds=n_extra_runs,
        offset_ds_in_exp=0,
        offset_exps=n_extra_experiments,
    )


def test_roundtrip_to_db_with_existing_meas_in_same_exp(
    two_empty_temp_db_connections, some_interdeps, tmp_path
) -> None:
    """
    Test that we can export from one db and import into another in
    the most basic form.
    """

    n_extra_runs = 4

    source_conn, target_conn = two_empty_temp_db_connections

    source_exp_1 = Experiment(conn=source_conn)
    target_exp_1 = Experiment(conn=target_conn)

    source_path = path_to_dbfile(source_conn)
    target_path = path_to_dbfile(target_conn)

    # make 5 runs in first experiment
    source_datasets = []

    # write datasets to test on

    for i in range(5):
        source_dataset = DataSetInMem._create_new_run(
            name=f"ds{i}",
            exp_id=source_exp_1.exp_id,
            path_to_db=source_path,
        )
        source_datasets.append(source_dataset)
        source_dataset._set_interdependencies(some_interdeps[1])

        source_dataset._perform_start_actions()

        for val in range(10):
            source_dataset._enqueue_results(
                {name: np.array([val]) for name in some_interdeps[1].names}
            )
        source_dataset.mark_completed()
        source_dataset.export(export_type="netcdf", path=str(tmp_path))

    extra_datasets = []

    for i in range(n_extra_runs):
        extra_dataset = DataSet(conn=target_conn, exp_id=target_exp_1.exp_id)

        extra_datasets.append(extra_dataset)
        extra_dataset.set_interdependencies(some_interdeps[1])

        extra_dataset.mark_started()

        for val in range(10):
            extra_dataset._enqueue_results(
                {name: np.array([val]) for name in some_interdeps[1].names}
            )
        extra_dataset.mark_completed()

    loaded_datasets = [
        load_from_netcdf(
            json.loads(ds.metadata["export_info"])["export_paths"]["nc"], target_path
        )
        for ds in source_datasets
    ]

    _assert_before_writing_metadata(source_datasets, loaded_datasets)

    for loaded_ds in loaded_datasets:
        loaded_ds.write_metadata_to_db()

    reloaded_ds = [load_by_guid(ds.guid, target_conn) for ds in loaded_datasets]

    _assert_after_writing_metadata(
        source_datasets,
        loaded_datasets,
        reloaded_ds,
        offset_total_ds=n_extra_runs,
        offset_ds_in_exp=n_extra_runs,
        offset_exps=0,
    )


def _assert_before_writing_metadata(source_datasets, loaded_datasets) -> None:
    for ds, lds in zip(source_datasets, loaded_datasets):
        # these should always be equal
        assert ds.guid == lds.guid

        assert ds.captured_run_id == lds.captured_run_id

        assert ds.captured_counter == lds.captured_counter

        assert ds.exp_name == lds.exp_name

        assert ds.sample_name == lds.sample_name

        # these are effectively counters and should be equal
        # in this case since we are inserting all runs in order
        # into an empty db

        # when loaded run_id/counter will be the captured run_id/counter e.g.
        # the original run_id/counter
        # after writing metadata to db it will be updated to match
        # see below
        assert ds.run_id == lds.run_id
        assert ds.counter == lds.counter

        # at this stage the exp_id is not set since
        # the dataset has not been written to a db
        # and exp_id is not a persistent attribute
        # this will be updated to the exp_id of the experiment
        # in the target db once written to the db
        assert lds.exp_id == 0


def _assert_after_writing_metadata(
    source_datasets,
    loaded_datasets,
    reloaded_ds,
    offset_total_ds,
    offset_ds_in_exp,
    offset_exps,
) -> None:
    for ds, lds, rlds in zip(source_datasets, loaded_datasets, reloaded_ds):
        # these should always be equal
        assert ds.guid == lds.guid
        assert ds.guid == rlds.guid

        assert ds.captured_run_id == lds.captured_run_id
        assert ds.captured_run_id == rlds.captured_run_id

        assert ds.captured_counter == lds.captured_counter
        assert ds.captured_counter == rlds.captured_counter

        assert ds.exp_name == lds.exp_name
        assert ds.exp_name == rlds.exp_name

        assert ds.sample_name == lds.sample_name
        assert ds.exp_name == rlds.exp_name

        # these are effectively counters and should be offset
        # since we are inserting all runs in order into a db with
        # existing runs and experiment

        assert ds.exp_id + offset_exps == lds.exp_id
        assert ds.exp_id + offset_exps == rlds.exp_id

        assert ds.run_id + offset_total_ds == lds.run_id
        assert ds.run_id + offset_total_ds == rlds.run_id

        assert ds.counter + offset_ds_in_exp == lds.counter
        assert ds.counter + offset_ds_in_exp == rlds.counter
