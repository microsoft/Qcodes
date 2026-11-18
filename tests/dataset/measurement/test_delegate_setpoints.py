from __future__ import annotations

import numpy as np
import pytest

from qcodes.dataset import Measurement
from qcodes.parameters import DelegateParameter, Parameter, ParameterWithSetpoints
from qcodes.validators import Arrays


def _make_params() -> tuple[Parameter, ParameterWithSetpoints]:
    p = Parameter("p", vals=Arrays(shape=(10,)), get_cmd=lambda: np.arange(10))
    ps1 = ParameterWithSetpoints(
        "ps1", vals=p.vals, setpoints=(p,), get_cmd=lambda: np.linspace(0, 1, 10)
    )
    return p, ps1


def test_delegate_forwards_depends_on_of_source() -> None:
    p, ps1 = _make_params()
    dp = DelegateParameter("dp", source=ps1)
    assert list(dp.depends_on) == [p]

    dp.source = None
    assert list(dp.depends_on) == []


def test_delegate_as_setpoint_does_not_hide_chained_dependency() -> None:
    # A setpoint that itself has setpoints is not supported. Before this fix,
    # hiding it behind a DelegateParameter silently dropped "p" from the dataset.
    _, ps1 = _make_params()
    dp = DelegateParameter("dp", source=ps1)
    ps2 = ParameterWithSetpoints(
        "ps2", vals=ps1.vals, setpoints=(dp,), get_cmd=lambda: np.zeros(10)
    )

    meas = Measurement()
    with pytest.raises(ValueError, match="both depends on"):
        meas.register_parameter(ps2)


@pytest.mark.usefixtures("experiment")
def test_measure_delegate_of_parameter_with_setpoints() -> None:
    _, ps1 = _make_params()
    dp = DelegateParameter("dp", source=ps1, vals=ps1.vals)

    meas = Measurement()
    meas.register_parameter(dp)
    assert set(meas.parameters) == {"dp", "p"}

    with meas.run() as datasaver:
        datasaver.add_result((dp, dp()))

    data = datasaver.dataset.get_parameter_data()
    np.testing.assert_array_equal(data["dp"]["p"][0], np.arange(10))
    np.testing.assert_array_equal(data["dp"]["dp"][0], np.linspace(0, 1, 10))
