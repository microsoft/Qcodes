"""
Tests for the export_datasets_and_create_metadata_db functionality
"""
import os
import tempfile
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


@pytest.fixture(name="simple_dataset")
def _simple_dataset():
    """Create a simple dataset for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        
        # Create experiment and dataset
        exp = load_or_create_experiment(
            experiment_name="test_exp",
            sample_name="test_sample",
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
        for i in range(10):
            dataset.add_results([{"x": i, "y": i**2}])
        
        dataset.mark_completed()
        
        yield db_path, dataset.run_id


def test_export_datasets_and_create_metadata_db_basic(simple_dataset):
    """Test basic functionality of export_datasets_and_create_metadata_db"""
    source_db_path, run_id = simple_dataset
    
    with tempfile.TemporaryDirectory() as temp_dir:
        target_db_path = Path(temp_dir) / "target.db"
        export_path = Path(temp_dir) / "exports"
        
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


def test_export_datasets_preserve_experiment_structure():
    """Test that experiment structure is preserved in the target database"""
    with tempfile.TemporaryDirectory() as temp_dir:
        source_db_path = Path(temp_dir) / "source.db"
        target_db_path = Path(temp_dir) / "target.db"
        export_path = Path(temp_dir) / "exports"
        
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


def test_export_datasets_with_incomplete_dataset():
    """Test behavior when source database contains incomplete datasets"""
    with tempfile.TemporaryDirectory() as temp_dir:
        source_db_path = Path(temp_dir) / "source.db"
        target_db_path = Path(temp_dir) / "target.db"
        export_path = Path(temp_dir) / "exports"
        
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


def test_export_datasets_empty_database():
    """Test behavior with empty source database"""
    with tempfile.TemporaryDirectory() as temp_dir:
        source_db_path = Path(temp_dir) / "empty.db"
        target_db_path = Path(temp_dir) / "target.db"
        export_path = Path(temp_dir) / "exports"
        
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


def test_export_datasets_default_export_path(simple_dataset):
    """Test that default export path is used when none provided"""
    source_db_path, run_id = simple_dataset
    
    with tempfile.TemporaryDirectory() as temp_dir:
        target_db_path = Path(temp_dir) / "target.db"
        
        # Run the export function without explicit export path
        result = export_datasets_and_create_metadata_db(
            source_db_path=source_db_path,
            target_db_path=target_db_path,
            # export_path=None  # Use default
        )
        
        # Should still work
        assert isinstance(result, dict)
        assert run_id in result


@pytest.mark.parametrize(
    "upgrade_source,upgrade_target",
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ]
)
def test_export_datasets_upgrade_flags(simple_dataset, upgrade_source, upgrade_target):
    """Test the upgrade flags functionality"""
    source_db_path, run_id = simple_dataset
    
    with tempfile.TemporaryDirectory() as temp_dir:
        target_db_path = Path(temp_dir) / "target.db"
        export_path = Path(temp_dir) / "exports"
        
        # Run the export function with different upgrade flags
        result = export_datasets_and_create_metadata_db(
            source_db_path=source_db_path,
            target_db_path=target_db_path,
            export_path=export_path,
            upgrade_source_db=upgrade_source,
            upgrade_target_db=upgrade_target,
        )
        
        # Function should complete successfully regardless of upgrade flags
        # (assuming databases are already current version)
        assert isinstance(result, dict)