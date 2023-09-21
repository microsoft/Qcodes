from functools import partial

import numpy as np
import pytest

from qcodes.dataset import do0d, do1d
from qcodes.parameters import ManualParameter, Parameter, ParameterWithSetpoints
from qcodes.validators import Arrays


@pytest.mark.usefixtures("experiment")
def test_register_name_with_manual_parameters():
    foo = ManualParameter("foo", initial_value=1, register_name="bar")
    foo_meas = ManualParameter("foo_meas", initial_value=2, register_name="bar_meas")

    ds, _, _ = do1d(foo, 0, 1, 101, 0, foo_meas)
    paramspecs = ds.get_parameters()
    assert len(paramspecs) == 2
    assert paramspecs[0].name in ("bar", "bar_meas")
    assert paramspecs[1].name in ("bar", "bar_meas")


@pytest.mark.usefixtures("experiment")
def test_register_name_with_pws():
    foo_setpoints = Parameter(
        name="foo_setpoints",
        get_cmd=partial(np.linspace, 0, 1, 101),
        vals=Arrays(shape=(101,)),
        register_name="bar_setpoints",
    )
    foo_meas = ParameterWithSetpoints(
        "foo_meas",
        setpoints=(foo_setpoints,),
        get_cmd=partial(np.linspace, 0, -1, 101),
        vals=Arrays(
            shape=(101,),
            valid_types=[np.integer, np.floating, np.complexfloating],
        ),
        register_name="bar_meas",
    )

    ds, _, _ = do0d(foo_meas)
    paramspecs = ds.get_parameters()
    assert len(paramspecs) == 2
    assert paramspecs[0].name in ("bar_setpoints", "bar_meas")
    assert paramspecs[1].name in ("bar_setpoints", "bar_meas")
