"""
Test for the inferred parameters fix.

Tests that inferred parameters are properly collected when adding results
to a dataset, even when they are transitively related through dependencies.
"""
import pytest
import tempfile
from pathlib import Path

from qcodes.dataset.sqlite.database import initialise_or_create_database_at
from qcodes.dataset.experiment_container import new_experiment  
from qcodes.dataset.measurements import Measurement
from qcodes.parameters import Parameter, DelegateParameter
from qcodes.instrument_drivers.mock_instruments import DummyInstrument
from qcodes.dataset.descriptions.dependencies import InterDependencies_
from qcodes.dataset.descriptions.param_spec import ParamSpecBase


def test_inferred_parameters_transitively_collected():
    """
    Test that parameters inferred from dependencies are properly collected
    when enqueuing results.
    """
    
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        initialise_or_create_database_at(db_path)
        
        # Create experiment  
        exp = new_experiment("test_exp", sample_name="test_sample")
        
        # Create mock instruments
        dac = DummyInstrument("dac", gates=["ch1", "ch2"])
        
        # Create delegate parameter that should be inferred from dac.ch1
        del_param = DelegateParameter("del_param_1", label="del param 1", source=dac.ch1)
        
        # Create a measurement parameter that depends on the delegate parameter
        measurement_param = Parameter("measurement", get_cmd=lambda: 42.0)
        
        # Create measurement
        meas = Measurement(name="test_measurement", exp=exp)
        
        # Register parameters to create the dependency chain:
        # measurement depends on del_param_1, del_param_1 is inferred from dac_ch1
        meas.register_parameter(dac.ch1)  # standalone
        meas.register_parameter(del_param, basis=(dac.ch1,))  # inferred from dac_ch1
        meas.register_parameter(measurement_param, setpoints=(del_param,))  # depends on del_param_1
        
        # Verify the interdependencies are set up correctly
        interdeps = meas._interdeps
        
        # Check that we have the expected structure
        assert len(interdeps.dependencies) == 1  # measurement depends on del_param_1
        assert len(interdeps.inferences) == 1    # del_param_1 inferred from dac_ch1
        assert len(interdeps.standalones) == 1   # dac_ch1 is standalone
        
        # Get the parameter specs
        measurement_spec = interdeps._id_to_paramspec["measurement"]
        del_param_spec = interdeps._id_to_paramspec["del_param_1"] 
        dac_spec = interdeps._id_to_paramspec["dac_ch1"]
        
        # Test the _collect_all_related_parameters method directly
        from qcodes.dataset.data_set import DataSet
        
        # Create a dummy dataset to access the method
        with meas.run() as datasaver:
            dataset = datasaver.dataset
            
            # Simulate a result_dict that would be passed to _enqueue_results
            result_dict = {
                measurement_spec: [1.0],
                del_param_spec: [0.5],
                dac_spec: [0.1]
            }
            
            # Test the helper method
            initial_params = {measurement_spec, del_param_spec}
            collected = dataset._collect_all_related_parameters(interdeps, initial_params, result_dict)
            
            # Verify that all three parameters are collected
            collected_names = {p.name for p in collected}
            expected_names = {"measurement", "del_param_1", "dac_ch1"}
            assert collected_names == expected_names, f"Expected {expected_names}, got {collected_names}"


def test_inferred_parameters_in_actual_measurement():
    """
    Test the full measurement flow to ensure inferred parameters are saved correctly.
    """
    
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        initialise_or_create_database_at(db_path)
        
        # Create experiment  
        exp = new_experiment("test_exp", sample_name="test_sample")
        
        # Create mock instruments
        dac = DummyInstrument("dac", gates=["ch1"])
        
        # Create delegate parameter
        del_param = DelegateParameter("del_param_1", label="del param 1", source=dac.ch1)
        
        # Create a standalone parameter to test that standalone handling still works
        standalone_param = Parameter("standalone", get_cmd=lambda: 3.14)
        
        # Create measurement
        meas = Measurement(name="test_measurement", exp=exp)
        
        # Register parameters  
        meas.register_parameter(dac.ch1)  # This should be standalone
        meas.register_parameter(del_param, basis=(dac.ch1,))  # This should be inferred from dac.ch1
        meas.register_parameter(standalone_param)  # This should be standalone
        
        # Run measurement
        with meas.run() as datasaver:
            # Set values and add results
            dac.ch1.set(0.5)
            del_param.set(0.5) 
            standalone_param.set(3.14)
            
            datasaver.add_result(
                (dac.ch1, dac.ch1()),
                (del_param, del_param()),
                (standalone_param, standalone_param()),
            )
            
        # Retrieve the dataset
        dataset = datasaver.dataset
        
        # Get parameter data - all parameters should be present
        param_data = dataset.get_parameter_data()
        
        # All parameters should be in the dataset
        assert "dac_ch1" in param_data, "dac_ch1 should be in parameter data"
        assert "del_param_1" in param_data, "del_param_1 should be in parameter data"  
        assert "standalone" in param_data, "standalone parameter should be in parameter data"
        
        # Check that the data is correct
        assert len(param_data["dac_ch1"]["dac_ch1"]) == 1
        assert len(param_data["del_param_1"]["del_param_1"]) == 1
        assert len(param_data["standalone"]["standalone"]) == 1


def test_multiple_dependent_parameters_no_cross_contamination():
    """
    Test that multiple dependent parameters that depend on the same independent
    parameter don't get mixed into each other's trees.
    
    This addresses the issue raised in comment_id 2167088469 where two dependent
    parameters (y1, y2) that both depend on the same independent parameter (x)
    would incorrectly get mixed into each other's parameter trees during
    _enqueue_results processing.
    """
    
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        initialise_or_create_database_at(db_path)
        
        # Create experiment  
        exp = new_experiment("test_exp", sample_name="test_sample")
        
        # Create independent parameter
        x_param = Parameter("x", get_cmd=lambda: 1.0, set_cmd=lambda val: None)
        
        # Create two dependent parameters that both depend on x
        y1_param = Parameter("y1", get_cmd=lambda: 2.0) 
        y2_param = Parameter("y2", get_cmd=lambda: 3.0)
        
        # Create measurement
        meas = Measurement(name="test_measurement", exp=exp)
        
        # Register parameters to create the problematic structure:
        # x (independent), y1 (depends on x), y2 (depends on x)
        meas.register_parameter(x_param)  # independent
        meas.register_parameter(y1_param, setpoints=(x_param,))  # y1 depends on x
        meas.register_parameter(y2_param, setpoints=(x_param,))  # y2 depends on x
        
        # Get the interdependencies and parameter specs
        interdeps = meas._interdeps
        x_spec = interdeps._id_to_paramspec["x"]
        y1_spec = interdeps._id_to_paramspec["y1"] 
        y2_spec = interdeps._id_to_paramspec["y2"]
        
        # Test the _collect_all_related_parameters method directly
        with meas.run() as datasaver:
            dataset = datasaver.dataset
            
            # Simulate a result_dict that would be passed to _enqueue_results
            result_dict = {
                x_spec: [1.0],
                y1_spec: [2.0],
                y2_spec: [3.0]
            }
            
            # Test collecting parameters for y1 tree
            # Should include x (its dependency) and y1, but NOT y2
            initial_params = {y1_spec}
            collected = dataset._collect_all_related_parameters(interdeps, initial_params, result_dict)
            
            collected_names = {p.name for p in collected}
            expected_names = {"x", "y1"}
            
            assert collected_names == expected_names, (
                f"y1 tree should only contain x and y1, but got {collected_names}. "
                f"This suggests y2 was incorrectly included due to cross-contamination."
            )
            
            # Test collecting parameters for y2 tree
            # Should include x (its dependency) and y2, but NOT y1
            initial_params = {y2_spec}
            collected = dataset._collect_all_related_parameters(interdeps, initial_params, result_dict)
            
            collected_names = {p.name for p in collected}
            expected_names = {"x", "y2"}
            
            assert collected_names == expected_names, (
                f"y2 tree should only contain x and y2, but got {collected_names}. "
                f"This suggests y1 was incorrectly included due to cross-contamination."
            )