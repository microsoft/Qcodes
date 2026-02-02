"""
Test for the inferred parameters fix.

Tests that inferred parameters are properly collected when adding results
to a dataset, even when they are transitively related through dependencies.
"""

from itertools import chain
from typing import TYPE_CHECKING

import numpy as np
import pytest

from qcodes.dataset import Measurement
from qcodes.dataset.descriptions.detect_shapes import detect_shape_of_measurement
from qcodes.dataset.experiment_container import Experiment
from qcodes.instrument_drivers.mock_instruments import DummyInstrument
from qcodes.parameters import DelegateParameter, ManualParameter, Parameter

if TYPE_CHECKING:
    from pytest import LogCaptureFixture

    from qcodes.dataset.experiment_container import Experiment
    from qcodes.instrument_drivers.mock_instruments import DummyInstrument


def test_inferred_parameters_transitively_collected(
    experiment: "Experiment", DAC: "DummyInstrument"
) -> None:
    """
    Test that parameters inferred from dependencies are properly collected
    when enqueuing results.
    """

    # Create delegate parameter that should be inferred from DAC.ch1
    del_param = DelegateParameter("del_param_1", label="del param 1", source=DAC.ch1)

    # Create a measurement parameter that depends on the delegate parameter
    measurement_param = Parameter("measurement", get_cmd=lambda: 42.0)

    # Create measurement
    meas = Measurement(name="test_measurement", exp=experiment)

    # Register parameters to create the dependency chain:
    # measurement depends on del_param_1, del_param_1 is inferred from dummy_dac_ch1
    meas.register_parameter(DAC.ch1)  # standalone
    meas.register_parameter(del_param, basis=(DAC.ch1,))  # inferred from dummy_dac_ch1
    meas.register_parameter(
        measurement_param, setpoints=(del_param,)
    )  # depends on del_param_1

    # Verify the interdependencies are set up correctly
    interdeps = meas._interdeps

    # Check that we have the expected structure
    assert len(interdeps.dependencies) == 1  # measurement depends on del_param_1
    assert len(interdeps.inferences) == 1  # del_param_1 inferred from dummy_dac_ch1
    assert len(interdeps.standalones) == 0  # there are no standalone parameters

    # Get the parameter specs
    measurement_spec = interdeps._id_to_paramspec["measurement"]
    del_param_spec = interdeps._id_to_paramspec["del_param_1"]
    dac_spec = interdeps._id_to_paramspec["dummy_dac_ch1"]

    # Test the collect_all_related_parameters method directly

    # Simulate a result_dict that would be passed to _enqueue_results
    result_dict = {measurement_spec: [1.0], del_param_spec: [0.5], dac_spec: [0.1]}

    # Test the helper method
    initial_params = {measurement_spec, del_param_spec}
    collected = set(
        chain.from_iterable(
            interdeps.find_all_parameters_in_tree(param) for param in initial_params
        )
    )
    # Filter to only include parameters that are in result_dict (same as the original behavior)
    collected = collected.intersection(result_dict.keys())

    # Verify that all three parameters are collected
    collected_names = {p.name for p in collected}
    expected_names = {"measurement", "del_param_1", "dummy_dac_ch1"}
    assert collected_names == expected_names, (
        f"Expected {expected_names}, got {collected_names}"
    )


def test_inferred_parameters_in_actual_measurement_0d(
    experiment: "Experiment", DAC: "DummyInstrument"
) -> None:
    """
    Test the full measurement flow to ensure inferred parameters are saved correctly.
    """

    # Create delegate parameter
    del_param = DelegateParameter("del_param_1", label="del param 1", source=DAC.ch1)

    # Create a standalone parameter to test that standalone handling still works
    standalone_param = ManualParameter("standalone")

    # Create measurement
    meas = Measurement(name="test_measurement", exp=experiment)

    # Register parameters
    meas.register_parameter(DAC.ch1)  # This should be standalone
    meas.register_parameter(
        del_param, basis=(DAC.ch1,)
    )  # This should be inferred from dac.ch1
    meas.register_parameter(standalone_param)  # This should be standalone

    # Run measurement
    with meas.run() as datasaver:
        # Set values and add results
        DAC.ch1.set(0.5)
        del_param.set(0.5)
        standalone_param.set(3.14)

        datasaver.add_result(
            (DAC.ch1, DAC.ch1()),
            (del_param, del_param()),
            (standalone_param, standalone_param()),
        )

    # Retrieve the dataset
    dataset = datasaver.dataset

    # Get parameter data - all parameters should be present
    param_data = dataset.get_parameter_data()

    # All top level parameters should be in the dataset
    assert "standalone" in param_data, (
        "standalone parameter should be in parameter data"
    )
    assert "del_param_1" in param_data, "del_param_1 should be in parameter data"

    # Check that the data is correct

    assert len(param_data["standalone"]["standalone"]) == 1
    assert len(param_data["del_param_1"]) == 2, (
        "del_param_1 tree should have two entries: del_param_1 and dummy_dac_ch1"
    )
    assert len(param_data["del_param_1"]["del_param_1"]) == 1
    assert len(param_data["del_param_1"]["dummy_dac_ch1"]) == 1


def test_inferred_parameters_in_actual_measurement_1d(
    experiment: "Experiment", DAC: "DummyInstrument"
) -> None:
    """
    Test the full measurement flow to ensure inferred parameters are saved correctly.
    """

    num_points = 10

    # Create delegate parameter
    del_param = DelegateParameter("del_param_1", label="del param 1", source=DAC.ch1)

    # Create a standalone parameter to test that standalone handling still works
    meas_parameter = ManualParameter("meas_parameter", initial_value=0.0)

    # Create measurement
    meas = Measurement(name="test_measurement", exp=experiment)

    # Register parameters
    meas.register_parameter(DAC.ch1)  # This should be standalone
    meas.register_parameter(
        del_param, basis=(DAC.ch1,)
    )  # This should be inferred from dac.ch1
    meas.register_parameter(
        meas_parameter, setpoints=(del_param,)
    )  # This should be standalone

    # Run measurement
    with meas.run() as datasaver:
        for i in np.linspace(0, 1, num_points):
            # Set values and add results
            del_param.set(i)

            datasaver.add_result(
                (DAC.ch1, DAC.ch1()),
                (del_param, del_param()),
                (meas_parameter, meas_parameter()),
            )

    # Retrieve the dataset
    dataset = datasaver.dataset

    # Get parameter data - all parameters should be present
    param_data = dataset.get_parameter_data()

    # All top level parameters should be in the dataset
    assert "meas_parameter" in param_data, "meas_parameter should be in parameter data"
    assert len(param_data) == 1

    meas_param_data = param_data["meas_parameter"]
    assert len(meas_param_data) == 3, (
        "meas_parameter tree should have three entries: meas_parameter, del_param_1, and dummy_dac_ch1"
    )
    assert len(meas_param_data["meas_parameter"]) == num_points, (
        "meas_parameter should have 10 entries for the 10 setpoints"
    )
    assert len(meas_param_data["del_param_1"]) == num_points, (
        "del_param_1 should have 10 entries for the 10 setpoints"
    )
    assert len(meas_param_data["dummy_dac_ch1"]) == num_points, (
        "dummy_dac_ch1 should have 10 entries for the 10 setpoints"
    )

    xarr = dataset.to_xarray_dataset()

    # Verify structure: the xarray dataset should have the meas_parameter as a data variable
    # and del_param_1 and dummy_dac_ch1 as coordinates
    assert "meas_parameter" in xarr.data_vars
    assert "del_param_1" in xarr.coords

    # inferred parameters are not currently exported
    assert "dummy_dac_ch1" not in xarr.coords

    # Check dimensions
    assert xarr.meas_parameter.dims == ("del_param_1",)
    assert len(xarr.meas_parameter.values) == num_points

    # Test export to pandas
    df = dataset.to_pandas_dataframe()

    # Check that all parameters are present in the DataFrame
    assert "meas_parameter" in df.columns
    assert df.index.name == "del_param_1"
    assert "dummy_dac_ch1" not in df.columns

    # Check that we have the right number of rows
    assert len(df) == num_points


@pytest.mark.parametrize("set_shape", [True, False])
def test_inferred_parameters_in_actual_measurement_2d(
    experiment: "Experiment",
    DAC: "DummyInstrument",
    caplog: "LogCaptureFixture",
    set_shape: bool,
) -> None:
    """
    2D version: both axes are DelegateParameters inferred from DAC channels.
    Ensures inferred parameters on both axes are saved and exported correctly.
    """
    num_points_x = 5
    num_points_y = 7

    # Delegate parameters (axes)
    del_param_1 = DelegateParameter("del_param_1", label="del param 1", source=DAC.ch1)
    del_param_2 = DelegateParameter("del_param_2", label="del param 2", source=DAC.ch2)

    # Measurement parameter depending on both delegate parameters
    meas_parameter = ManualParameter("meas_parameter", initial_value=0.0)

    meas = Measurement(name="test_measurement_2d", exp=experiment)

    # Register underlying basis (standalone) parameters
    meas.register_parameter(DAC.ch1)
    meas.register_parameter(DAC.ch2)

    # Register delegate parameters inferred from their respective DAC channels
    meas.register_parameter(del_param_1, basis=(DAC.ch1,))
    meas.register_parameter(del_param_2, basis=(DAC.ch2,))

    # Register measurement parameter with 2D setpoints
    meas.register_parameter(meas_parameter, setpoints=(del_param_1, del_param_2))
    if set_shape:
        meas.set_shapes(
            detect_shape_of_measurement([meas_parameter], (num_points_x, num_points_y))
        )

    with meas.run() as datasaver:
        for x in np.linspace(0, 1, num_points_x):
            for y in np.linspace(0, 2, num_points_y):
                del_param_1.set(x)
                del_param_2.set(y)
                meas_parameter.set(x + y)  # simple deterministic value

                datasaver.add_result(
                    (DAC.ch1, DAC.ch1()),
                    (DAC.ch2, DAC.ch2()),
                    (del_param_1, del_param_1()),
                    (del_param_2, del_param_2()),
                    (meas_parameter, meas_parameter()),
                )

    dataset = datasaver.dataset
    param_data = dataset.get_parameter_data()

    assert "meas_parameter" in param_data
    assert len(param_data) == 1  # only one top-level dependent parameter tree

    tree = param_data["meas_parameter"]
    expected_keys = {
        "meas_parameter",
        "del_param_1",
        "del_param_2",
        "dummy_dac_ch1",
        "dummy_dac_ch2",
    }
    assert set(tree.keys()) == expected_keys

    total_points = num_points_x * num_points_y
    for k in expected_keys:
        assert len(tree[k].ravel()) == total_points, (
            f"{k} should have {total_points} entries"
        )

    # xarray export
    caplog.clear()
    with caplog.at_level("INFO"):
        xarr = dataset.to_xarray_dataset()

    assert len(caplog.records) == 1
    if set_shape:
        assert (
            caplog.records[0].message
            == "Exporting meas_parameter to xarray using direct method"
        )
    else:
        assert (
            caplog.records[0].message
            == "Exporting meas_parameter to xarray via pandas index"
        )

    assert "meas_parameter" in xarr.data_vars

    assert "del_param_1" in xarr.coords
    assert xarr.coords["del_param_1"].shape == (num_points_x,)
    assert xarr.coords["del_param_1"].dims == ("del_param_1",)

    assert "del_param_2" in xarr.coords
    assert xarr.coords["del_param_2"].shape == (num_points_y,)
    assert xarr.coords["del_param_2"].dims == ("del_param_2",)

    if set_shape:
        assert "dummy_dac_ch1" in xarr.coords
        assert xarr.coords["dummy_dac_ch1"].shape == (num_points_x,)
        assert xarr.coords["dummy_dac_ch1"].dims == ("del_param_1",)

        assert "dummy_dac_ch2" in xarr.coords
        assert xarr.coords["dummy_dac_ch2"].shape == (num_points_y,)
        assert xarr.coords["dummy_dac_ch2"].dims == ("del_param_2",)
    else:
        assert "dummy_dac_ch1" not in xarr.coords
        assert "dummy_dac_ch2" not in xarr.coords
    assert xarr["meas_parameter"].dims == ("del_param_1", "del_param_2")
    assert xarr["meas_parameter"].shape == (num_points_x, num_points_y)

    # pandas export
    df = dataset.to_pandas_dataframe()
    assert "meas_parameter" in df.columns
    assert df.index.names == ["del_param_1", "del_param_2"]
    assert "dummy_dac_ch1" not in df.columns
    assert "dummy_dac_ch2" not in df.columns
    assert len(df) == total_points


def test_multiple_dependent_parameters_no_cross_contamination(
    experiment: "Experiment",
) -> None:
    """
    Test that multiple dependent parameters that depend on the same independent
    parameter don't get mixed into each other's trees.

    This addresses the issue raised in comment_id 2167088469 where two dependent
    parameters (y1, y2) that both depend on the same independent parameter (x)
    would incorrectly get mixed into each other's parameter trees during
    _enqueue_results processing.
    """

    # Create independent parameter
    x_param = Parameter("x", get_cmd=lambda: 1.0, set_cmd=lambda val: None)

    # Create two dependent parameters that both depend on x
    y1_param = Parameter("y1", get_cmd=lambda: 2.0)
    y2_param = Parameter("y2", get_cmd=lambda: 3.0)

    # Create measurement
    meas = Measurement(name="test_measurement", exp=experiment)

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

    # Test the collect_all_related_parameters method directly

    # Simulate a result_dict that would be passed to _enqueue_results
    result_dict = {x_spec: [1.0], y1_spec: [2.0], y2_spec: [3.0]}

    # Test collecting parameters for y1 tree
    # Should include x (its dependency) and y1, but NOT y2
    initial_params = {y1_spec}
    collected = set(
        chain.from_iterable(
            interdeps.find_all_parameters_in_tree(param) for param in initial_params
        )
    )
    # Filter to only include parameters that are in result_dict (same as the original behavior)
    collected = collected.intersection(result_dict.keys())

    collected_names = {p.name for p in collected}
    expected_names = {"x", "y1"}

    assert collected_names == expected_names, (
        f"y1 tree should only contain x and y1, but got {collected_names}. "
        f"This suggests y2 was incorrectly included due to cross-contamination."
    )

    # Test collecting parameters for y2 tree
    # Should include x (its dependency) and y2, but NOT y1
    initial_params = {y2_spec}
    collected = set(
        chain.from_iterable(
            interdeps.find_all_parameters_in_tree(param) for param in initial_params
        )
    )
    # Filter to only include parameters that are in result_dict (same as the original behavior)
    collected = collected.intersection(result_dict.keys())

    collected_names = {p.name for p in collected}
    expected_names = {"x", "y2"}

    assert collected_names == expected_names, (
        f"y2 tree should only contain x and y2, but got {collected_names}. "
        f"This suggests y1 was incorrectly included due to cross-contamination."
    )
