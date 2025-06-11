"""
Tests for the export_datasets_and_create_metadata_db functionality
"""
from pathlib import Path

import pytest

from qcodes.dataset import (
    DataSet,
    Experiment,
    export_datasets_and_create_metadata_db,
    load_by_id,
    load_or_create_experiment,
)
from qcodes.dataset.descriptions.dependencies import InterDependencies_
from qcodes.dataset.descriptions.param_spec import ParamSpec
from qcodes.dataset.sqlite.connection import connect
from qcodes.dataset.sqlite.queries import get_runs


@pytest.fixture
def dataset_factory():
    """Factory fixture for creating datasets with configurable parameters"""
    def _create_dataset(tmp_path, name="test_dataset", exp_name="test_exp", sample_name="test_sample", num_points=10):
        db_path = tmp_path / f"{name}.db"
        
        # Create experiment and dataset
        exp = load_or_create_experiment(
            experiment_name=exp_name,
            sample_name=sample_name,
            conn=connect(db_path)
        )
        
        # Create interdependencies
        x = ParamSpec("x", "numeric", unit="V")
        y = ParamSpec("y", "numeric", unit="A")
        interdeps = InterDependencies_(dependencies={y: (x,)})
        
        # Create dataset
        dataset = DataSet(conn=exp.conn, exp_id=exp.exp_id)
        dataset.set_interdependencies(interdeps)
        dataset.mark_started()
        
        # Add some data
        for i in range(num_points):
            dataset.add_results([{"x": i, "y": i**2}])
        
        dataset.mark_completed()
        
        return db_path, dataset.run_id, dataset
    
    return _create_dataset


@pytest.fixture
def simple_dataset(tmp_path, dataset_factory):
    """Create a simple dataset for testing"""
    return dataset_factory(tmp_path)


def test_export_datasets_and_create_metadata_db_basic(tmp_path, simple_dataset):
    """Test basic functionality of export_datasets_and_create_metadata_db"""
    source_db_path, run_id, _ = simple_dataset
    
    target_db_path = tmp_path / "target.db"
    export_path = tmp_path / "exports"
    
    # Run the export function
    result = export_datasets_and_create_metadata_db(
        source_db_path=source_db_path,
        target_db_path=target_db_path,
        export_path=export_path,
    )
    
    # Check that the function returned a result
    assert isinstance(result, dict)
    assert run_id in result
    assert result[run_id] in ["exported", "copied_as_is"]
    
    # Check that target database was created
    assert target_db_path.exists()
    
    # Check that target database has the run
    target_conn = connect(target_db_path)
    target_runs = get_runs(target_conn)
    assert len(target_runs) == 1
    target_conn.close()
    
    # Check that NetCDF file was created if export was successful
    if result[run_id] == "exported":
        netcdf_files = list(export_path.glob("*.nc"))
        assert len(netcdf_files) > 0


def test_export_datasets_preserve_experiment_structure(tmp_path):
    """Test that experiment structure is preserved in the target database"""
    source_db_path = tmp_path / "source.db"
    target_db_path = tmp_path / "target.db"
    export_path = tmp_path / "exports"
    
    # Create source database with multiple experiments
    source_conn = connect(source_db_path)
    
    # Create first experiment
    exp1 = load_or_create_experiment(
        experiment_name="exp1",
        sample_name="sample1",
        conn=source_conn
    )
    
    # Create second experiment
    exp2 = load_or_create_experiment(
        experiment_name="exp2",
        sample_name="sample2",
        conn=source_conn
    )
    
    # Create interdependencies
    x = ParamSpec("x", "numeric", unit="V")
    y = ParamSpec("y", "numeric", unit="A")
    interdeps = InterDependencies_(dependencies={y: (x,)})
    
    # Create datasets in both experiments
    datasets = []
    for exp in [exp1, exp2]:
        for i in range(2):  # 2 datasets per experiment
            dataset = DataSet(conn=source_conn, exp_id=exp.exp_id)
            dataset.set_interdependencies(interdeps)
            dataset.mark_started()
            
            # Add some data
            for j in range(5):
                dataset.add_results([{"x": j, "y": j * (i + 1)}])
            
            dataset.mark_completed()
            datasets.append(dataset)
    
    source_conn.close()
    
    # Run the export function
    result = export_datasets_and_create_metadata_db(
        source_db_path=source_db_path,
        target_db_path=target_db_path,
        export_path=export_path,
    )
    
    # Check that all datasets were processed
    assert len(result) == 4
    
    # Check that target database has all runs
    target_conn = connect(target_db_path)
    target_runs = get_runs(target_conn)
    assert len(target_runs) == 4
    target_conn.close()


def test_export_datasets_with_incomplete_dataset(tmp_path):
    """Test behavior when source database contains incomplete datasets"""
    source_db_path = tmp_path / "source.db"
    target_db_path = tmp_path / "target.db"
    export_path = tmp_path / "exports"
    
    # Create source database
    source_conn = connect(source_db_path)
    exp = load_or_create_experiment(
        experiment_name="test_exp",
        sample_name="test_sample",
        conn=source_conn
    )
    
    # Create interdependencies
    x = ParamSpec("x", "numeric", unit="V")
    y = ParamSpec("y", "numeric", unit="A")
    interdeps = InterDependencies_(dependencies={y: (x,)})
    
    # Create completed dataset
    dataset1 = DataSet(conn=source_conn, exp_id=exp.exp_id)
    dataset1.set_interdependencies(interdeps)
    dataset1.mark_started()
    for i in range(5):
        dataset1.add_results([{"x": i, "y": i**2}])
    dataset1.mark_completed()
    
    # Create incomplete dataset
    dataset2 = DataSet(conn=source_conn, exp_id=exp.exp_id)
    dataset2.set_interdependencies(interdeps)
    dataset2.mark_started()
    for i in range(3):
        dataset2.add_results([{"x": i, "y": i**3}])
    # Note: not marking as completed
    
    source_conn.close()
    
    # Run the export function
    result = export_datasets_and_create_metadata_db(
        source_db_path=source_db_path,
        target_db_path=target_db_path,
        export_path=export_path,
    )
    
    # Check that both datasets were processed
    assert len(result) == 2
    assert dataset1.run_id in result
    assert dataset2.run_id in result
    
    # Incomplete dataset should be copied as-is
    assert result[dataset2.run_id] == "copied_as_is"


def test_export_datasets_empty_database(tmp_path):
    """Test behavior with empty source database"""
    source_db_path = tmp_path / "empty.db"
    target_db_path = tmp_path / "target.db"
    export_path = tmp_path / "exports"
    
    # Create empty database
    source_conn = connect(source_db_path)
    source_conn.close()
    
    # Run the export function
    result = export_datasets_and_create_metadata_db(
        source_db_path=source_db_path,
        target_db_path=target_db_path,
        export_path=export_path,
    )
    
    # Should return empty result
    assert result == {}


def test_export_datasets_default_export_path(tmp_path, simple_dataset):
    """Test that default export path is used when none provided"""
    source_db_path, run_id, _ = simple_dataset
    
    target_db_path = tmp_path / "target.db"
    
    # Run the export function without explicit export path
    result = export_datasets_and_create_metadata_db(
        source_db_path=source_db_path,
        target_db_path=target_db_path,
        # export_path=None  # Use default
    )
    
    # Should still work
    assert isinstance(result, dict)
    assert run_id in result


def test_export_datasets_handles_export_failure(tmp_path):
    """Test that the function handles export failures gracefully"""
    source_db_path = tmp_path / "source.db"
    target_db_path = tmp_path / "target.db"
    export_path = tmp_path / "exports"
    
    # Create source database
    source_conn = connect(source_db_path)
    exp = load_or_create_experiment(
        experiment_name="test_exp",
        sample_name="test_sample",
        conn=source_conn
    )
    
    # Create interdependencies with problematic data that might fail export
    x = ParamSpec("x", "text", unit="")  # Text data might be harder to export
    y = ParamSpec("y", "numeric", unit="A")
    interdeps = InterDependencies_(dependencies={y: (x,)})
    
    # Create dataset with mixed data types
    dataset = DataSet(conn=source_conn, exp_id=exp.exp_id)
    dataset.set_interdependencies(interdeps)
    dataset.mark_started()
    
    # Add some data that might be challenging to export
    for i in range(3):
        dataset.add_results([{"x": f"text_{i}", "y": i**2}])
    
    dataset.mark_completed()
    source_conn.close()
    
    # Run the export function
    result = export_datasets_and_create_metadata_db(
        source_db_path=source_db_path,
        target_db_path=target_db_path,
        export_path=export_path,
    )
    
    # Should handle the dataset one way or another
    assert len(result) == 1
    assert dataset.run_id in result
    # Should either export or copy as-is
    assert result[dataset.run_id] in ["exported", "copied_as_is"]


def test_export_datasets_nonexistent_source(tmp_path):
    """Test behavior with non-existent source database"""
    source_db_path = tmp_path / "nonexistent.db"
    target_db_path = tmp_path / "target.db"
    export_path = tmp_path / "exports"
    
    # Should handle non-existent source gracefully
    with pytest.raises((FileNotFoundError, OSError)):
        export_datasets_and_create_metadata_db(
            source_db_path=source_db_path,
            target_db_path=target_db_path,
            export_path=export_path,
        )


def test_export_datasets_readonly_target(tmp_path, simple_dataset):
    """Test behavior when target path is not writable"""
    source_db_path, run_id, _ = simple_dataset
    
    # Create a read-only directory for target
    readonly_dir = tmp_path / "readonly"
    readonly_dir.mkdir()
    readonly_dir.chmod(0o444)  # Read-only
    
    try:
        target_db_path = readonly_dir / "target.db"
        export_path = tmp_path / "exports"
        
        # Should handle permission errors gracefully
        with pytest.raises((PermissionError, OSError)):
            export_datasets_and_create_metadata_db(
                source_db_path=source_db_path,
                target_db_path=target_db_path,
                export_path=export_path,
            )
    finally:
        # Restore permissions for cleanup
        readonly_dir.chmod(0o755)


def test_export_datasets_large_dataset_scenario(tmp_path):
    """Test handling of a scenario with multiple datasets including edge cases"""
    source_db_path = tmp_path / "source.db"
    target_db_path = tmp_path / "target.db"
    export_path = tmp_path / "exports"
    
    # Create source database
    source_conn = connect(source_db_path)
    exp = load_or_create_experiment(
        experiment_name="test_exp",
        sample_name="test_sample",
        conn=source_conn
    )
    
    # Create interdependencies
    x = ParamSpec("x", "numeric", unit="V")
    y = ParamSpec("y", "numeric", unit="A")
    interdeps = InterDependencies_(dependencies={y: (x,)})
    
    created_datasets = []
    
    # Create several datasets with different characteristics
    for i in range(5):
        dataset = DataSet(conn=source_conn, exp_id=exp.exp_id)
        dataset.set_interdependencies(interdeps)
        dataset.mark_started()
        
        # Add varying amounts of data
        for j in range(i + 1):  # 1, 2, 3, 4, 5 data points respectively
            dataset.add_results([{"x": j, "y": j * (i + 1)}])
        
        if i < 4:  # Leave one dataset incomplete
            dataset.mark_completed()
        
        created_datasets.append(dataset)
    
    source_conn.close()
    
    # Run the export function
    result = export_datasets_and_create_metadata_db(
        source_db_path=source_db_path,
        target_db_path=target_db_path,
        export_path=export_path,
    )
    
    # Check that all datasets were processed
    assert len(result) == 5
    
    # Check that target database has all runs
    target_conn = connect(target_db_path)
    target_runs = get_runs(target_conn)
    target_conn.close()
    
    assert len(target_runs) == 5
    
    # The incomplete dataset should be copied as-is
    incomplete_dataset = created_datasets[-1]
    assert result[incomplete_dataset.run_id] == "copied_as_is"


def test_export_datasets_prevents_overwriting_target(tmp_path, simple_dataset):
    """Test that the function prevents overwriting existing target database files"""
    source_db_path, run_id, _ = simple_dataset
    target_db_path = tmp_path / "target.db"
    export_path = tmp_path / "exports"
    
    # Create an existing target database file
    target_db_path.touch()
    
    # Should raise FileExistsError when target already exists
    with pytest.raises(FileExistsError, match="Target database file already exists"):
        export_datasets_and_create_metadata_db(
            source_db_path=source_db_path,
            target_db_path=target_db_path,
            export_path=export_path,
        )


def test_export_datasets_status_reporting(tmp_path):
    """Test that the function returns detailed status information"""
    source_db_path = tmp_path / "source.db"
    target_db_path = tmp_path / "target.db"
    export_path = tmp_path / "exports"
    
    # Create source database with a completed dataset
    source_conn = connect(source_db_path)
    exp = load_or_create_experiment(
        experiment_name="test_exp",
        sample_name="test_sample",
        conn=source_conn
    )
    
    # Create interdependencies
    x = ParamSpec("x", "numeric", unit="V")
    y = ParamSpec("y", "numeric", unit="A")
    interdeps = InterDependencies_(dependencies={y: (x,)})
    
    # Create and complete a dataset
    dataset = DataSet(conn=source_conn, exp_id=exp.exp_id)
    dataset.set_interdependencies(interdeps)
    dataset.mark_started()
    
    for i in range(5):
        dataset.add_results([{"x": i, "y": i**2}])
    
    dataset.mark_completed()
    source_conn.close()
    
    # Run the export function
    result = export_datasets_and_create_metadata_db(
        source_db_path=source_db_path,
        target_db_path=target_db_path,
        export_path=export_path,
    )
    
    # Check return value structure
    assert isinstance(result, dict)
    assert len(result) == 1
    assert dataset.run_id in result
    
    # Status should be one of the expected values
    status = result[dataset.run_id]
    expected_statuses = ["exported", "copied_as_is", "already_exists"]
    assert status in expected_statuses, f"Unexpected status: {status}"
    
    # If we run again, should report already_exists
    result2 = export_datasets_and_create_metadata_db(
        source_db_path=source_db_path,
        target_db_path=target_db_path,
        export_path=export_path,
    )
    
    assert result2[dataset.run_id] == "already_exists"