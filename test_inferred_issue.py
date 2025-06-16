#!/usr/bin/env python3
"""
Test script to reproduce the inferred parameters issue
"""
import tempfile
from pathlib import Path
import numpy as np

# Test the issue
def test_reproduce_inferred_params_issue():
    """Test that reproduces the issue where inferred parameters are missing from dataset"""
    
    # Import after setup
    from qcodes.dataset.sqlite.database import initialise_or_create_database_at
    from qcodes.dataset.experiment_container import new_experiment  
    from qcodes.dataset.measurements import Measurement
    from qcodes.parameters import Parameter, DelegateParameter
    from qcodes.instrument_drivers.mock_instruments import DummyInstrument
    
    # Create a temporary database with a unique name
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        initialise_or_create_database_at(db_path)
        
        # Create experiment  
        exp = new_experiment("test_exp", sample_name="test_sample")
        
        # Create mock instruments
        dac = DummyInstrument("dac", gates=["ch1", "ch2"])
        
        # Create delegate parameter that should be inferred
        del_param = DelegateParameter("del_param_1", label="del param 1", source=dac.ch1)
        
        # Create measurement
        meas = Measurement(name="1d_measurement", exp=exp)
        
        # Register parameters - this should create inference relationship
        meas.register_parameter(dac.ch1)  # This will be the basis
        meas.register_parameter(del_param, basis=(dac.ch1,))  # This should be inferred from dac.ch1
        
        # Check that the interdependencies are set up correctly
        interdeps = meas._interdeps
        print("Dependencies:", dict(interdeps.dependencies))
        print("Inferences:", dict(interdeps.inferences))
        
        # The issue is in _enqueue_results - let me simulate it here 
        # to show the problem
        
        # Simulate a result_dict that would be passed to _enqueue_results
        from qcodes.dataset.descriptions.param_spec import ParamSpecBase
        
        # Get the param specs from the measurement
        param_specs = list(interdeps._id_to_paramspec.values())
        dac_spec = None
        del_spec = None
        for ps in param_specs:
            if ps.name == "dac_ch1":
                dac_spec = ps
            elif ps.name == "del_param_1":
                del_spec = ps
        
        # Simulate result_dict that add_result would create
        result_dict = {
            dac_spec: np.array([0.1]),
            del_spec: np.array([0.1])
        }
        
        # Test the problematic logic from _enqueue_results
        toplevel_params = set(interdeps.dependencies).intersection(set(result_dict))
        print(f"Toplevel params: {[p.name for p in toplevel_params]}")
        
        # This is the problematic part
        for toplevel_param in toplevel_params:
            inff_params = set(interdeps.inferences.get(toplevel_param, ()))
            deps_params = set(interdeps.dependencies.get(toplevel_param, ()))
            all_params = inff_params.union(deps_params).union({toplevel_param})
            
            print(f"For toplevel param {toplevel_param.name}:")
            print(f"  Inff params: {[p.name for p in inff_params]}")
            print(f"  Deps params: {[p.name for p in deps_params]}")
            print(f"  All params: {[p.name for p in all_params]}")
            print(f"  Result dict keys: {[p.name for p in result_dict.keys()]}")
            
            # Check if del_param_1 is in result_dict but not in all_params
            result_param_names = {p.name for p in result_dict.keys()}
            all_param_names = {p.name for p in all_params}
            missing_params = result_param_names - all_param_names
            if missing_params:
                print(f"  PROBLEM: Missing params {missing_params} not collected!")
                return False
        
        print("Test passed - all inferred parameters would be collected")
        return True

if __name__ == "__main__":
    test_reproduce_inferred_params_issue()