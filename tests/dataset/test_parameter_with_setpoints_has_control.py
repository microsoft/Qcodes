from typing import TYPE_CHECKING

import numpy as np

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

    mp = ManualParameter("mp", vals=Arrays(shape=(10,)), initial_value=np.arange(10))
    p1 = ParameterWithSetpoints(
        "p1", vals=Arrays(shape=(10,)), setpoints=(mp,), set_cmd=None
    )
    p2 = MySp("p2", vals=Arrays(shape=(10,)), setpoints=(mp,), set_cmd=None)
    p2.has_control_of.add(p1)

    p1(np.linspace(-1, 1, 10))
    p2(np.random.randn(10))

    meas = Measurement()
    meas.register_parameter(p2)
    with meas.run() as ds:
        ds.add_result((p2, p2()))

    xds = ds.dataset.to_xarray_dataset()  # does not unravel to grid

    assert (
        list(xds.sizes.keys()) == ["mp"]
    )  # without p1 this correctly has mp as the only dim, with p1 this is turned into a generic 'index' dim
