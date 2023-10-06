from functools import partial

import numpy as np
import pytest

from qcodes.dataset import do0d, do1d
from qcodes.dataset.measurements import str_or_register_name
from qcodes.parameters import ManualParameter, Parameter, ParameterWithSetpoints
from qcodes.validators import Arrays


@pytest.mark.usefixtures("experiment")
def test_register_name_with_manual_parameters():
    indep_param = ManualParameter(
        "indep_param", initial_value=1, register_name="renamed_indep_param"
    )
    dep_param = ManualParameter(
        "dep_param", initial_value=2, register_name="renamed_dep_param"
    )

    ds, _, _ = do1d(indep_param, 0, 1, 101, 0, dep_param)
    paramspecs = ds.get_parameters()
    assert len(paramspecs) == 2
    assert paramspecs[0].name in ("renamed_indep_param", "renamed_dep_param")
    assert paramspecs[1].name in ("renamed_indep_param", "renamed_dep_param")


@pytest.mark.usefixtures("experiment")
def test_register_name_with_pws():
    setpoints_param = Parameter(
        name="setpoints_param",
        get_cmd=partial(np.linspace, 0, 1, 101),
        vals=Arrays(shape=(101,)),
        register_name="renamed_setpoints",
    )
    meas_param = ParameterWithSetpoints(
        "meas_param",
        setpoints=(setpoints_param,),
        get_cmd=partial(np.linspace, 0, -1, 101),
        vals=Arrays(
            shape=(101,),
            valid_types=[np.integer, np.floating, np.complexfloating],
        ),
        register_name="renamed_meas_param",
    )

    ds, _, _ = do0d(meas_param)
    paramspecs = ds.get_parameters()
    assert len(paramspecs) == 2
    assert paramspecs[0].name in ("renamed_setpoints", "renamed_meas_param")
    assert paramspecs[1].name in ("renamed_setpoints", "renamed_meas_param")


def test_register_name_utils():
    indep_param = ManualParameter(
        "indep_param", initial_value=1, register_name="renamed_indep_param"
    )

    assert indep_param.register_name == "renamed_indep_param"
    assert str_or_register_name(indep_param) == "renamed_indep_param"
    assert str_or_register_name(str(indep_param)) == "indep_param"
