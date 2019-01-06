from numpy.random import rand
import pytest


from qcodes.instrument.parameter import ParameterWithSetpoints, Parameter
import qcodes.utils.validators as vals





def test_verification_invalid_shape():
    """
    Test that various parameters with setpoints and invalid shape combinations
    raises when validated.
    """

    n_points_1 = Parameter('n_points_1', set_cmd=None, vals=vals.Ints())
    n_points_2 = Parameter('n_points_2', set_cmd=None, vals=vals.Ints())

    n_points_1.set(10)
    n_points_2.set(10)

    setpoints_1 = Parameter('setpoints_1', get_cmd=lambda: rand(n_points_1()),
                          vals=vals.Arrays(shape=(n_points_1,)))
    setpoints_2 = Parameter('setpoints_2', get_cmd=lambda: rand(n_points_2()),
                          vals=vals.Arrays(shape=(n_points_2,)))

    param_with_setpoints = ParameterWithSetpoints('param',
                                                  get_cmd=lambda:
                                                  rand(n_points_2()),
                                                  setpoints=(setpoints_1,),
                                                  vals=vals.Arrays(shape=(n_points_2,)))

    # the two shapes are the same so validation works
    param_with_setpoints.validate_consistent_shape()
    param_with_setpoints.validate(param_with_setpoints.get())

    n_points_2(20)
    with pytest.raises(ValueError):
        param_with_setpoints.validate_consistent_shape()
    with pytest.raises(ValueError):
        param_with_setpoints.validate(param_with_setpoints.get())
