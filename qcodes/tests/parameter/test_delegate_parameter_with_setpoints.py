import numpy as np
from numpy.random import rand
import pytest
from qcodes.dataset import DataSetProtocol

from qcodes.instrument.parameter import (
    DelegateParameter,
    ParameterWithSetpoints,
    Parameter,
    expand_setpoints_helper,
)
import qcodes.utils.validators as vals
from qcodes.utils.dataset.doNd import dond

from qcodes.instrument.parameter import DelegateParameterWithSetpoints


@pytest.fixture(name="parameters")
def _make_parameters():
    n_points_1 = Parameter("n_points_1", set_cmd=None, vals=vals.Ints())
    n_points_2 = Parameter("n_points_2", set_cmd=None, vals=vals.Ints())
    n_points_3 = Parameter("n_points_3", set_cmd=None, vals=vals.Ints())

    n_points_1.set(10)
    n_points_2.set(20)
    n_points_3.set(15)

    setpoints_1 = Parameter(
        "setpoints_1",
        get_cmd=lambda: np.arange(n_points_1()),
        vals=vals.Arrays(shape=(n_points_1,)),
    )
    setpoints_2 = Parameter(
        "setpoints_2",
        get_cmd=lambda: np.arange(n_points_2()),
        vals=vals.Arrays(shape=(n_points_2,)),
    )
    setpoints_3 = Parameter(
        "setpoints_3",
        get_cmd=lambda: np.arange(n_points_3()),
        vals=vals.Arrays(shape=(n_points_3,)),
    )
    yield (n_points_1, n_points_2, n_points_3, setpoints_1, setpoints_2, setpoints_3)


def test_validation_shapes():
    n_points_1 = Parameter("n_points_1", set_cmd=None, vals=vals.Ints())
    n_points_2 = Parameter("n_points_2", set_cmd=None, vals=vals.Ints())

    n_points_1.set(10)
    n_points_2.set(20)

    setpoints_1 = Parameter(
        "setpoints_1",
        get_cmd=lambda: rand(n_points_1()),
        vals=vals.Arrays(shape=(n_points_1,)),
    )
    setpoints_2 = Parameter(
        "setpoints_2",
        get_cmd=lambda: rand(n_points_2()),
        vals=vals.Arrays(shape=(n_points_2,)),
    )

    # 1D

    param_with_setpoints_1 = ParameterWithSetpoints(
        "param_1",
        get_cmd=lambda: rand(n_points_1()),
        setpoints=(setpoints_1,),
        vals=vals.Arrays(shape=(n_points_1,)),
    )

    delegate_param_1 = DelegateParameterWithSetpoints(
        "delegate_param_1",
        source=param_with_setpoints_1,
        new_setpoints=(DelegateParameter("delegate_stepoint_1", None),),
    )

    delegate_param_1.validate_consistent_shape()
    delegate_param_1.validate(delegate_param_1.get())

    # 2D

    param_with_setpoints_2 = ParameterWithSetpoints(
        "param_2",
        get_cmd=lambda: rand(n_points_1(), n_points_2()),
        setpoints=(setpoints_1, setpoints_2),
        vals=vals.Arrays(shape=(n_points_1, n_points_2)),
    )

    delegate_param_2 = DelegateParameterWithSetpoints(
        "delegate_param_2",
        source=param_with_setpoints_2,
        new_setpoints=(
            DelegateParameter("delegate_setpoint_1", None),
            DelegateParameter("delegate_setpoint_2", None),
        ),
    )

    delegate_param_2.validate_consistent_shape()
    delegate_param_2.validate(delegate_param_2.get())


def test_expand_setpoints_1d(parameters):
    """
    Test that the setpoints expander helper function works correctly
    """

    (
        n_points_1,
        n_points_2,
        n_points_3,
        setpoints_1,
        setpoints_2,
        setpoints_3,
    ) = parameters

    param_with_setpoints_1 = ParameterWithSetpoints(
        "param_1",
        get_cmd=lambda: rand(n_points_1()),
        setpoints=(setpoints_1,),
        vals=vals.Arrays(shape=(n_points_1,)),
    )

    delegate_param_1 = DelegateParameterWithSetpoints(
        "delegate_param_1",
        source=param_with_setpoints_1,
        new_setpoints=(DelegateParameter("delegate_setpoint_1", None),),
    )

    data = expand_setpoints_helper(delegate_param_1)

    assert len(data) == 2
    assert len(data[0][1]) == len(data[1][1])


def test_expand_setpoints_2d(parameters):

    (
        n_points_1,
        n_points_2,
        n_points_3,
        setpoints_1,
        setpoints_2,
        setpoints_3,
    ) = parameters

    param_with_setpoints_2 = ParameterWithSetpoints(
        "param_2",
        get_cmd=lambda: rand(n_points_1(), n_points_2()),
        vals=vals.Arrays(shape=(n_points_1, n_points_2)),
        setpoints=(setpoints_1, setpoints_2),
    )

    delegate_param_2 = DelegateParameterWithSetpoints(
        "delegate_param_2",
        source=param_with_setpoints_2,
        new_setpoints=(
            DelegateParameter("delegate_setpoint_1", None),
            DelegateParameter("delegate_setpoint_2", None),
        ),
    )

    data = expand_setpoints_helper(delegate_param_2)

    assert len(data) == 3
    assert data[0][1].shape == data[1][1].shape
    assert data[0][1].shape == data[2][1].shape

    sp1 = data[0][1]
    sp2 = data[1][1]
    # the first set of setpoints should be repeated along the second axis
    for i in range(sp1.shape[1]):
        np.testing.assert_array_equal(sp1[:, i], np.arange(sp1.shape[0]))
    # the second set of setpoints should be repeated along the first axis
    for i in range(sp2.shape[0]):
        np.testing.assert_array_equal(sp2[i, :], np.arange(sp1.shape[1]))


def test_delegate_parameter_with_setpoints_in_measurement(parameters, empty_experiment):
    _ = empty_experiment
    (
        n_points_1,
        n_points_2,
        n_points_3,
        setpoints_1,
        setpoints_2,
        setpoints_3,
    ) = parameters

    param_with_setpoints_1 = ParameterWithSetpoints(
        "param_1",
        get_cmd=lambda: rand(n_points_1()),
        setpoints=(setpoints_1,),
        vals=vals.Arrays(shape=(n_points_1,)),
    )

    delegate_param_1 = DelegateParameterWithSetpoints(
        "delegate_param_1",
        source=param_with_setpoints_1,
        new_setpoints=(DelegateParameter("delegate_setpoint_1", None),),
    )

    ds, _, _ = dond(delegate_param_1)
    assert isinstance(ds, DataSetProtocol)

    all_param_names = set(ds.description.interdeps.names)
    assert all_param_names == {delegate_param_1.name, "delegate_setpoint_1"}
