from typing import TYPE_CHECKING

import numpy as np
import numpy.testing as npt

from qcodes.dataset import Measurement
from qcodes.parameters import ManualParameter, ParameterWithSetpoints
from qcodes.validators import Arrays

if TYPE_CHECKING:
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
    p2_data = np.random.randn(10)
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
