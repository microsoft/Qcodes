import logging
from typing import TYPE_CHECKING

import numpy as np
import numpy.testing as npt
import xarray as xr

from qcodes.dataset import Measurement
from qcodes.dataset.exporters.export_to_xarray import _add_inferred_data_vars
from qcodes.parameters import ManualParameter, ParameterWithSetpoints
from qcodes.validators import Arrays

if TYPE_CHECKING:
    import pytest

    from qcodes.dataset.experiment_container import Experiment


def test_parameter_with_setpoints_has_control(experiment: "Experiment"):
    class MySp(ParameterWithSetpoints):
        def unpack_self(self, value):
            res = super().unpack_self(value)
            res.append((p1, p1()))
            return res

    mp_data = np.arange(10)
    p1_data = np.linspace(-1, 1, 10)

    mp = ManualParameter("mp", vals=Arrays(shape=(10,)), initial_value=mp_data)
    p1 = ParameterWithSetpoints(
        "p1", vals=Arrays(shape=(10,)), setpoints=(mp,), set_cmd=None
    )
    p2 = MySp("p2", vals=Arrays(shape=(10,)), setpoints=(mp,), set_cmd=None)
    p2.has_control_of.add(p1)

    p1(p1_data)
    p2_data = np.random.default_rng().standard_normal(10)
    p2(p2_data)

    meas = Measurement()
    meas.register_parameter(p2)

    # Only p2 should be top-level; p1 is inferred from p2
    interdeps = meas._interdeps
    top_level_names = [p.name for p in interdeps.top_level_parameters]
    assert top_level_names == ["p2"]

    with meas.run() as ds:
        ds.add_result((p2, p2()))

    # Verify raw parameter data has exactly one row per parameter
    raw_data = ds.dataset.get_parameter_data()
    assert list(raw_data.keys()) == ["p2"], "Only p2 should be a top-level result"
    for name, arr in raw_data["p2"].items():
        assert arr.shape == (1, 10), (
            f"Expected shape (1, 10) for {name}, got {arr.shape}"
        )

    xds = ds.dataset.to_xarray_dataset()

    # mp should be the only dimension (not a generic 'index')
    assert list(xds.sizes.keys()) == ["mp"]
    assert xds.sizes["mp"] == 10

    # mp values used as coordinate axis
    npt.assert_array_equal(xds.coords["mp"].values, mp_data)

    # p2 is the primary data variable with correct values
    assert "p2" in xds.data_vars
    npt.assert_array_almost_equal(xds["p2"].values, p2_data)

    # p1 is included as a data variable (inferred from p2) with correct values
    assert "p1" in xds.data_vars
    npt.assert_array_almost_equal(xds["p1"].values, p1_data)

    # p1 data is also retrievable from the raw parameter data
    npt.assert_array_almost_equal(raw_data["p2"]["p1"].ravel(), p1_data)


def test_parameter_with_setpoints_has_control_2d(experiment: "Experiment"):
    """Test that an inferred parameter with the same size as its parent
    but different from the full dimension product is correctly included."""

    class MySp(ParameterWithSetpoints):
        def unpack_self(self, value):
            res = super().unpack_self(value)
            res.append((p1, p1()))
            return res

    n_x = 3
    n_y = 4
    mp_x_data = np.arange(n_x, dtype=float)
    mp_y_data = np.arange(n_y, dtype=float)

    mp_x = ManualParameter("mp_x", initial_value=0.0)
    mp_y = ManualParameter("mp_y", vals=Arrays(shape=(n_y,)), initial_value=mp_y_data)

    p1 = ParameterWithSetpoints(
        "p1", vals=Arrays(shape=(n_y,)), setpoints=(mp_y,), set_cmd=None
    )
    p2 = MySp("p2", vals=Arrays(shape=(n_y,)), setpoints=(mp_y,), set_cmd=None)
    p2.has_control_of.add(p1)

    meas = Measurement()
    meas.register_parameter(p2, setpoints=(mp_x,))

    p1_all = []
    p2_all = []

    with meas.run() as ds:
        for x_val in mp_x_data:
            mp_x(x_val)
            p1_row = np.linspace(-1, 1, n_y) + x_val
            p1(p1_row)
            p2_row = np.random.default_rng().standard_normal(n_y)
            p2(p2_row)
            p1_all.append(p1_row)
            p2_all.append(p2_row)
            ds.add_result((mp_x, mp_x()), (p2, p2()))

    p1_all_arr = np.array(p1_all)
    p2_all_arr = np.array(p2_all)

    xds = ds.dataset.to_xarray_dataset()

    # Should have 2 dimensions: mp_x and mp_y
    assert set(xds.sizes.keys()) == {"mp_x", "mp_y"}
    assert xds.sizes["mp_x"] == n_x
    assert xds.sizes["mp_y"] == n_y

    # p2 is the primary data variable
    assert "p2" in xds.data_vars
    npt.assert_array_almost_equal(xds["p2"].values, p2_all_arr)

    # p1 is included as a data variable (inferred from p2)
    # Its size (n_x * n_y = 12) matches its parent p2's size,
    # which differs from either individual dimension.
    assert "p1" in xds.data_vars
    npt.assert_array_almost_equal(xds["p1"].values, p1_all_arr)


def test_parameter_with_setpoints_has_control_size_mismatch_warns(
    experiment: "Experiment", caplog: "pytest.LogCaptureFixture"
) -> None:
    """Test that a warning is emitted when the inferred parameter has a
    different data size than its parent parameter."""

    class MySp(ParameterWithSetpoints):
        def unpack_self(self, value):
            res = super().unpack_self(value)
            res.append((p1, p1()))
            return res

    mp_data = np.arange(10)

    mp = ManualParameter("mp", vals=Arrays(shape=(10,)), initial_value=mp_data)
    p1 = ParameterWithSetpoints(
        "p1", vals=Arrays(shape=(10,)), setpoints=(mp,), set_cmd=None
    )
    p2 = MySp("p2", vals=Arrays(shape=(10,)), setpoints=(mp,), set_cmd=None)
    p2.has_control_of.add(p1)

    p1(np.linspace(-1, 1, 10))
    p2(np.random.default_rng().standard_normal(10))

    meas = Measurement()
    meas.register_parameter(p2)
    with meas.run() as ds:
        ds.add_result((p2, p2()))

    # Build an xarray dataset and sub_dict with mismatched p1 data to
    # exercise the warning path in _add_inferred_data_vars directly.

    raw_data = ds.dataset.get_parameter_data()
    sub_dict = dict(raw_data["p2"])
    # Replace p1 with wrong-sized data (5 instead of 10)
    sub_dict["p1"] = np.zeros(5)

    xr_dataset = xr.Dataset(
        {"p2": (("mp",), sub_dict["p2"].ravel())},
        coords={"mp": sub_dict["mp"].ravel()},
    )

    with caplog.at_level(
        logging.WARNING, logger="qcodes.dataset.exporters.export_to_xarray"
    ):
        result = _add_inferred_data_vars(ds.dataset, "p2", sub_dict, xr_dataset)

    assert "p1" not in result.data_vars
    assert any(
        "Cannot add inferred parameter 'p1'" in msg and "'p2'" in msg
        for msg in caplog.messages
    )
